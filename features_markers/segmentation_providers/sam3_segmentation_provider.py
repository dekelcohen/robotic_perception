from __future__ import annotations

from typing import List, Any

from .segmentation_providers import ISegmentation
from .roboflow_helper import create_roboflow_client, add_np_masks_to_results


class SAM3SegmentationProvider(ISegmentation):
    """Segmentation using RoboFlow's SAM3 workflow."""

    def __init__(
        self,
        workspace_name: str = "dekel-cohen",
        workflow_id: str = "sam3",
        api_url: str = "https://serverless.roboflow.com",
        api_key: str | None = None,
        use_cache: bool = True,
        client: Any | None = None,
    ) -> None:
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.api_key = api_key
        self.use_cache = use_cache
        self._client = client

    def _client_lazy(self):
        if self._client is None:
            self._client = create_roboflow_client(api_url=self.api_url, api_key=self.api_key)
        return self._client

    def segment(self, image_or_path, prompt_classes: List[str]) -> Any:
        client = self._client_lazy()

        # Determine payload: path or in-memory
        images_payload = {"image": image_or_path}

        # Minimal observability without introducing a logger dependency
        try:
            src_desc = image_or_path if isinstance(image_or_path, str) else f"{type(image_or_path).__name__}"
        except Exception:
            src_desc = "<image>"
        print(f"SAM3: running workflow='{self.workflow_id}' on image='{src_desc}' with classes={prompt_classes}")

        response = client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images=images_payload,
            parameters={"classes": list(prompt_classes) if prompt_classes else []},
            use_cache=self.use_cache,
        )
        return add_np_masks_to_results(response)
