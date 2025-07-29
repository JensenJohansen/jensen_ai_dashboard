from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import *
from .serializers import *
from .vanna import VannaService
from .superset import SupersetClient

class IsOwnerOrOrgUser(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return (obj.user == request.user) or (
            hasattr(request.user, 'organization') and
            obj.user.organization == request.user.organization
        )

class DatabaseInstanceViewSet(viewsets.ModelViewSet):
    queryset = DatabaseInstance.objects.all()
    serializer_class = DatabaseInstanceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return DatabaseInstance.objects.filter(created_by=user) | DatabaseInstance.objects.filter(created_by__organization=user.organization)

    @action(detail=True, methods=['post'], url_path='train')
    def train(self, request, pk=None):
        db_instance = self.get_object()
        model_name = request.data.get("model", "grok")
        service = VannaService(user=request.user, model_name=model_name)
        try:
            service.train_model(db_instance)
            return Response({"detail": "Training complete."})
        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'], url_path='prompt')
    def prompt(self, request, pk=None):
        db_instance = self.get_object()
        prompt = request.data.get("prompt")
        model_name = request.data.get("model", "grok")
        if not prompt:
            return Response({"detail": "Prompt is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            service = VannaService(user=request.user, model_name=model_name)
            result = service.process_prompt(prompt=prompt, db_instance=db_instance)
            return Response(result)
        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class QueryViewSet(viewsets.ModelViewSet):
    queryset = Query.objects.all()
    serializer_class = QuerySerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrOrgUser]

    def get_queryset(self):
        user = self.request.user
        return Query.objects.filter(user=user) | Query.objects.filter(user__organization=user.organization)

class DashboardViewSet(viewsets.ModelViewSet):
    queryset = Dashboard.objects.all()
    serializer_class = DashboardSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrOrgUser]

    def get_queryset(self):
        user = self.request.user
        return Dashboard.objects.filter(user=user) | Dashboard.objects.filter(user__organization=user.organization)

    @action(detail=True, methods=['post'], url_path='generate')
    def generate(self, request, pk=None):
        dashboard = self.get_object()
        query_ids = request.data.get("query_ids", [])
        if not query_ids:
            return Response({"detail": "query_ids is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            queries = Query.objects.filter(id__in=query_ids)
            superset = SupersetClient()
            superset_dashboard_id = superset.create_dashboard(dashboard.title)

            for query in queries:
                dataset_id = superset.create_dataset(query.table_name, database_id=query.db_instance.superset_db_id)
                chart_id = superset.create_chart(dataset_id, chart_type="table", chart_name=query.title)
                superset.add_chart_to_dashboard(superset_dashboard_id, chart_id)

            dashboard.superset_id = superset_dashboard_id
            dashboard.superset_url = superset.get_embedded_dashboard_url(superset_dashboard_id)
            dashboard.save()

            return Response({"detail": "Superset dashboard created.", "url": dashboard.superset_url})
        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['get'], url_path='describe')
    def describe(self, request, pk=None):
        dashboard = self.get_object()
        queries = dashboard.queries.all()
        description = {
            "dashboard_title": dashboard.title,
            "superset_url": dashboard.superset_url,
            "queries": [
                {
                    "title": q.title,
                    "sql": q.sql,
                    "prompt": q.prompt,
                    "table_name": q.table_name
                } for q in queries
            ]
        }
        return Response(description)
    
    @action(detail=True, methods=['post'], url_path='explain-and-generate')
    def explain_and_generate(self, request, pk=None):
        dashboard = self.get_object()
        db_instance = dashboard.db_instance  # Ensure FK exists
        prompt = request.data.get("prompt")
        model_name = request.data.get("model", "grok")

        if not prompt:
            return Response({"detail": "Prompt is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            service = VannaService(user=request.user, model_name=model_name)
            results = service.generate_queries_from_description(prompt=prompt, db_instance=db_instance)
            superset = SupersetClient()
            superset_dashboard_id = superset.create_dashboard(dashboard.title)
            created_queries = []

            for result in results:
                query = Query.objects.create(
                    user=request.user,
                    db_instance=db_instance,
                    sql=result["sql"],
                    prompt=result["prompt"],
                    title=result.get("title", "Generated Query"),
                    table_name=result.get("table", "unknown")
                )
                dashboard.queries.add(query)
                dataset_id = superset.create_dataset(query.table_name, database_id=query.db_instance.superset_db_id)
                chart_id = superset.create_chart(dataset_id, chart_type="table", chart_name=query.title)
                superset.add_chart_to_dashboard(superset_dashboard_id, chart_id)
                created_queries.append({"id": query.id, "title": query.title})

            dashboard.superset_id = superset_dashboard_id
            dashboard.superset_url = superset.get_embedded_dashboard_url(superset_dashboard_id)
            dashboard.save()

            return Response({
                "detail": "Dashboard and queries generated.",
                "queries": created_queries,
                "superset_url": dashboard.superset_url
            })
        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PromptLogViewSet(viewsets.ModelViewSet):
    queryset = PromptLog.objects.all()
    serializer_class = PromptLogSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return PromptLog.objects.filter(user=user) | PromptLog.objects.filter(user__organization=user.organization)
