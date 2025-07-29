import requests
from django.conf import settings

class SupersetClient:
    def __init__(self):
        self.base_url = settings.SUPERSET_BASE_URL.rstrip('/')
        self.username = settings.SUPERSET_USERNAME
        self.password = settings.SUPERSET_PASSWORD
        self.access_token = self.login()

    def login(self):
        url = f"{self.base_url}/api/v1/security/login"
        payload = {
            "username": self.username,
            "password": self.password,
            "provider": "db"
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("access_token")

    def headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def create_dataset(self, table_name, database_id, schema=None):
        url = f"{self.base_url}/api/v1/dataset"
        payload = {
            "database": database_id,
            "table_name": table_name,
        }
        if schema:
            payload["schema"] = schema

        response = requests.post(url, json=payload, headers=self.headers())
        response.raise_for_status()
        return response.json()["id"]

    def create_chart(self, dataset_id, chart_type="table", chart_name="Generated Chart"):
        url = f"{self.base_url}/api/v1/chart/"
        payload = {
            "slice_name": chart_name,
            "viz_type": chart_type,
            "datasource_id": dataset_id,
            "datasource_type": "table"
        }
        response = requests.post(url, json=payload, headers=self.headers())
        response.raise_for_status()
        return response.json()["id"]

    def create_dashboard(self, dashboard_title):
        url = f"{self.base_url}/api/v1/dashboard"
        payload = {"dashboard_title": dashboard_title}
        response = requests.post(url, json=payload, headers=self.headers())
        response.raise_for_status()
        return response.json()["id"]

    def add_chart_to_dashboard(self, dashboard_id, chart_id):
        url = f"{self.base_url}/api/v1/dashboard/{dashboard_id}"
        response = requests.get(url, headers=self.headers())
        response.raise_for_status()
        dashboard_data = response.json()["result"]

        positions = dashboard_data.get("position_json", {})
        new_position = {
            "type": "CHART",
            "meta": {"chartId": chart_id},
            "children": [],
            "id": f"CHART-{chart_id}",
            "parents": ["ROOT_ID"]
        }
        positions[f"CHART-{chart_id}"] = new_position

        payload = {
            "position_json": positions,
            "json_metadata": dashboard_data.get("json_metadata", "")
        }
        response = requests.put(url, json=payload, headers=self.headers())
        response.raise_for_status()
        return response.json()

    def get_embedded_dashboard_url(self, dashboard_id):
        return f"{self.base_url}/superset/dashboard/{dashboard_id}/?standalone=1"
