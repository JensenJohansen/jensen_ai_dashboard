import logging
import os
import pandas as pd
import re
import sqlparse
from traceback import format_exc
from typing import Optional, List, Dict, Any
from django.db import connection
from django.utils.timezone import now
from django.contrib.auth import get_user_model
from common.storage import upload_directory_to_minio
from core.models import DatabaseInstance, Query, PromptLog
from .vanna_backends.grok import GrokVanna
from .vanna_backends.openai import OpenAIVanna
from .vanna_backends.ollama3 import OllamaVanna


# Monkey-patch LogRecord factory to add default 'user_id' if missing
_old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    if 'user_id' not in record.__dict__:
        record.user_id = 'None'
    return record

logging.setLogRecordFactory(record_factory)

# Configure file-based logging
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "vanna_service.log"),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] User=%(user_id)s %(message)s'
)

logger = logging.getLogger(__name__)

User = get_user_model()

MODEL_MAP = {
    "grok": GrokVanna,
    "openai": OpenAIVanna,
    "ollama": OllamaVanna
}

class VannaService:
    def __init__(self, user: User, model_name: str = "grok", allow_llm_data: bool = False):
        """Initialize VannaService with user, model, and privacy settings."""
        if not user:
            logger.error("User cannot be None", extra={'user_id': 'N/A'})
            raise ValueError("User is required")
        if model_name.lower() not in MODEL_MAP:
            logger.error(f"Invalid model name: {model_name}", extra={'user_id': user.id})
            raise ValueError(f"Model {model_name} is not supported. Choose from {list(MODEL_MAP.keys())}")
        self.user = user
        self.allow_llm_data = allow_llm_data
        backend_class = MODEL_MAP.get(model_name.lower(), GrokVanna)
        self.vn = backend_class(config={"model": model_name})
        self.vn.persist_path = f"./vanna_storage/{self.user.id}/"
        logger.info(f"Initialized VannaService with model {model_name}", extra={'user_id': user.id})

    def describe_database(self, db_instance: DatabaseInstance) -> List:
        """Return all table descriptions for a database instance."""
        if not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid db_instance provided", extra={'user_id': self.user.id})
            raise ValueError("db_instance must be a DatabaseInstance object")
        return list(db_instance.tabledescription_set.all())

    def log_prompt(self, prompt: str, db_instance: DatabaseInstance, status: str = "success", error: str = "") -> PromptLog:
        """Log prompt to file and database."""
        if not prompt or not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid prompt or db_instance", extra={'user_id': self.user.id})
            raise ValueError("Prompt and valid db_instance are required")
        logger.info(f"Prompt='{prompt}' Status={status} Error={error}", extra={'user_id': self.user.id})
        return PromptLog.objects.create(
            user=self.user,
            database=db_instance,
            prompt=prompt,
            status=status,
            error_message=error,
            created_at=now()
        )

    def extract_table_name(self, sql: str) -> str:
        """Extract table name from SQL query using sqlparse."""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                logger.warning(f"No valid SQL parsed: {sql}", extra={'user_id': self.user.id})
                return ""
            for statement in parsed:
                for token in statement.tokens:
                    if isinstance(token, sqlparse.sql.Identifier) and token.get_parent_name():
                        return f"{token.get_parent_name()}.{token.get_name()}"
                    elif isinstance(token, sqlparse.sql.Identifier):
                        return token.get_name()
            logger.warning(f"No table name found in SQL: {sql}", extra={'user_id': self.user.id})
            return ""
        except Exception as e:
            logger.error(f"Error extracting table name from SQL: {sql}, Error: {str(e)}", extra={'user_id': self.user.id})
            return ""

    def extract_keywords(self, prompt: str) -> List[str]:
        """Extract keywords from prompt, filtering out short words."""
        if not prompt or not isinstance(prompt, str):
            logger.error("Invalid prompt for keyword extraction", extra={'user_id': self.user.id})
            raise ValueError("Prompt must be a non-empty string")
        keywords = [word for word in prompt.lower().split() if len(word) > 3]
        logger.debug(f"Extracted keywords: {keywords}", extra={'user_id': self.user.id})
        return keywords

    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame by handling NaN, timedeltas, and mixed types."""
        if df is None or df.empty:
            logger.warning("Empty or None DataFrame provided", extra={'user_id': self.user.id})
            return df
        df = df.where(pd.notnull(df), None)
        for col in df.select_dtypes(include=['timedelta']):
            df[col] = df[col].dt.total_seconds()
        for col in df.columns:
            if df[col].dropna().map(type).nunique() > 1:
                logger.warning(f"Converting mixed-type column {col} to string", extra={'user_id': self.user.id})
                df[col] = df[col].astype(str)
        return df

    def calculate_confidence(self, prompt: str, sql: str, df: Optional[pd.DataFrame]) -> float:
        """Calculate confidence score for a generated query."""
        try:
            similar = self.vn.get_similar_question_sql(prompt, n=1)
            similarity = 1.0 - similar[0].get('distance', 1.0) if similar and isinstance(similar, list) else 0.0
            sql_valid = 1.0 if re.match(r'^(select|with)\b', sql, re.IGNORECASE) else 0.5
            result_score = min(len(df) / 10.0, 1.0) if df is not None and not df.empty else 0.0
            confidence = (0.5 * similarity) + (0.3 * sql_valid) + (0.2 * result_score)
            confidence = round(min(max(confidence, 0.0), 1.0), 2)
            logger.debug(f"Confidence calculated: similarity={similarity}, sql_valid={sql_valid}, result_score={result_score}, total={confidence}", extra={'user_id': self.user.id})
            return confidence
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}", extra={'user_id': self.user.id})
            return 0.0

    def generate_query(self, db_instance: DatabaseInstance, prompt: str) -> Query:
        """Generate and save a SQL query from a prompt."""
        if not prompt or not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid prompt or db_instance", extra={'user_id': self.user.id})
            raise ValueError("Prompt and valid db_instance are required")
        try:
            sql, df, _ = self.vn.ask(prompt, print_results=False, allow_llm_to_see_data=self.allow_llm_data)
            if not sql:
                raise ValueError("No SQL query generated")
            query = Query.objects.create(
                user=self.user,
                db_instance=db_instance,
                prompt=prompt,
                sql=sql,
                success=True
            )
            self.log_prompt(prompt, db_instance, status="success")
            return query
        except Exception as e:
            logger.exception(f"Failed to generate query: {str(e)}", extra={'user_id': self.user.id})
            self.log_prompt(prompt, db_instance, status="error", error=str(e))
            raise

    def test_query_execution(self, query: Query) -> List[Any]:
        """Test execution of a SQL query."""
        if not isinstance(query, Query) or not query.sql:
            logger.error("Invalid query or empty SQL", extra={'user_id': self.user.id})
            raise ValueError("Valid Query object with SQL is required")
        try:
            with connection.cursor() as cursor:
                cursor.execute(query.sql)
                # Use fetchmany to handle large result sets
                results = []
                while True:
                    batch = cursor.fetchmany(size=1000)
                    if not batch:
                        break
                    results.extend(batch)
                return results
        except Exception as e:
            logger.exception(f"Query execution failed: {str(e)}", extra={'user_id': self.user.id})
            raise

    def process_prompt(self, prompt: str, db_instance: DatabaseInstance) -> Dict[str, Any]:
        """Process a prompt to generate SQL, execute it, and return results."""
        if not prompt or not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid prompt or db_instance", extra={'user_id': self.user.id})
            raise ValueError("Prompt and valid db_instance are required")
        try:
            vector_store_path = os.path.join(self.vn.persist_path, str(db_instance.id))
            self.vn.persist_path = vector_store_path
            if self.allow_llm_data:
                logger.warning("Allowing LLM to see data, potential privacy risk", extra={'user_id': self.user.id})
            sql, df, _ = self.vn.ask(prompt, print_results=False, allow_llm_to_see_data=self.allow_llm_data)
            if not sql:
                raise ValueError("No SQL query generated")
            df = self.sanitize_dataframe(df)
            query = Query.objects.create(
                user=self.user,
                db_instance=db_instance,
                prompt=prompt,
                sql=sql,
                success=True
            )
            self.log_prompt(prompt, db_instance, status="success")
            return {
                "sql_query": sql,
                "results": df.to_dict(orient='records') if df is not None and not df.empty else [],
                "detected_table": self.extract_table_name(sql),
                "keywords": self.extract_keywords(prompt),
                "confidence": self.calculate_confidence(prompt, sql, df)
            }
        except Exception as e:
            logger.exception(f"Failed to process prompt: {str(e)}", extra={'user_id': self.user.id})
            self.log_prompt(prompt, db_instance, status="error", error=str(e))
            raise

    def train_model(self, db_instance: DatabaseInstance) -> bool:
        """Train the Vanna model with database schema and upload to MinIO."""
        if not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid db_instance", extra={'user_id': self.user.id})
            raise ValueError("db_instance must be a DatabaseInstance object")
        try:
            vector_store_path = os.path.join(self.vn.persist_path, str(db_instance.id))
            self.vn.persist_path = vector_store_path

            # Database-specific schema query
            db_type = connection.vendor
            if db_type == 'postgresql':
                schema_sql = "SELECT * FROM information_schema.columns"
            elif db_type == 'mysql':
                schema_sql = "SELECT * FROM information_schema.columns"
            elif db_type == 'sqlite':
                schema_sql = "SELECT * FROM sqlite_master WHERE type='table'"
            else:
                logger.error(f"Unsupported database type: {db_type}", extra={'user_id': self.user.id})
                raise ValueError(f"Unsupported database type: {db_type}")

            df_schema = self.vn.run_sql(schema_sql)
            plan = self.vn.get_training_plan_generic(df_schema)
            self.vn.train(plan=plan)

            # Upload to MinIO
            object_path = f"vanna-training/{self.user.id}/{db_instance.id}/"
            upload_directory_to_minio(vector_store_path, object_path)

            db_instance.is_trained = True
            db_instance.last_trained_at = now()
            db_instance.trained_by = self.user
            db_instance.trained_model_path = object_path
            db_instance.save()

            logger.info(f"Training completed for database {db_instance.id}", extra={'user_id': self.user.id})
            return True
        except Exception as e:
            logger.error(f"Training failed for database {db_instance.id}: {str(e)}", extra={'user_id': self.user.id})
            raise

    def generate_queries_from_description(self, prompt: str, db_instance: DatabaseInstance) -> List[Dict[str, Any]]:
        """Generate query suggestions from a prompt."""
        if not prompt or not isinstance(db_instance, DatabaseInstance):
            logger.error("Invalid prompt or db_instance", extra={'user_id': self.user.id})
            raise ValueError("Prompt and valid db_instance are required")
        try:
            self.vn.connect(db_instance)
            suggestions = self.vn.generate_query_suggestions(prompt)[:10]  # Limit to 10 suggestions
            results = []
            for suggestion in suggestions:
                sql = suggestion.get("sql", "")
                try:
                    df = self.vn.run_sql(sql) if sql else None
                    results.append({
                        "prompt": suggestion.get("prompt", prompt),
                        "sql": sql,
                        "title": suggestion.get("title", "Generated Query"),
                        "table": self.extract_table_name(sql),
                        "confidence": self.calculate_confidence(prompt, sql, df)
                    })
                except Exception as e:
                    logger.warning(f"Failed to execute suggestion SQL: {sql}, Error: {str(e)}", extra={'user_id': self.user.id})
                    results.append({
                        "prompt": suggestion.get("prompt", prompt),
                        "sql": sql,
                        "title": suggestion.get("title", "Generated Query"),
                        "table": self.extract_table_name(sql),
                        "confidence": 0.0
                    })
            return results
        except Exception as e:
            logger.error(f"Error generating queries from description: {str(e)}", extra={'user_id': self.user.id})
            raise