"""AI 赛道投研多智能体平台。"""
from .config import Settings
from .pipeline import AnalysisResult, CompanyInput, SectorPipeline as ExpandedSectorPipeline, SectorRequest, save_partial_result, save_result
from .main_pipeline import MainAgentPipeline
from .tools import DocumentStore

SectorPipeline = MainAgentPipeline

__all__ = ["Settings", "SectorPipeline", "MainAgentPipeline", "ExpandedSectorPipeline", "SectorRequest", "CompanyInput", "AnalysisResult", "DocumentStore", "save_result", "save_partial_result"]
