import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "gout_llmops.serving.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
