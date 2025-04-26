from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入新增的路由器
from .routers import agents, analysis, api_runs, logs, runs, workflow

# Create FastAPI app instance
app = FastAPI(
    title="A Share Investment Agent - Backend",
    description="API for monitoring LLM interactions within the agent workflow.",
    version="0.1.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
# Allows requests from any origin in this example.
# Adjust origins as needed for production environments.
origins = ["*"]  # Allow all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# 包含现有路由器
app.include_router(logs.router)
app.include_router(runs.router)

# 包含新增的路由器
app.include_router(agents.router)
app.include_router(workflow.router)
app.include_router(analysis.router)
app.include_router(api_runs.router)

# 根端点API导航


@app.get("/")
def read_root():
    return {
        "message": "欢迎使用A股投资Agent后端API! 访问 /docs 了解详情。",
        "api_navigation": {
            "文档": "/docs",
            "新API": {
                "介绍": "采用标准化的ApiResponse格式的新API",
                "端点": {
                    "代理": "/api/agents/",
                    "分析": "/api/analysis/",
                    "运行": "/api/runs/",
                    "工作流": "/api/workflow/",
                },
            },
            "旧API": {
                "介绍": "为向后兼容保留的原有API",
                "端点": {"日志": "/logs/", "运行": "/runs/"},
            },
        },
    }


@app.get("/api")
def api_navigation():
    """提供API导航信息"""
    return {
        "message": "A股投资Agent API导航",
        "api_sections": {
            "/api/agents": "获取各个Agent的状态和数据",
            "/api/analysis": "启动和查询股票分析任务",
            "/api/runs": "查询运行历史和状态(基于api_state)",
            "/api/workflow": "获取当前工作流状态",
        },
        "legacy_api": {
            "/logs": "查询历史LLM交互日志",
            "/runs": "详细查询运行历史和Agent执行数据(基于BaseLogStorage)",
        },
        "documentation": {"OpenAPI文档": "/docs", "ReDoc文档": "/redoc"},
    }
