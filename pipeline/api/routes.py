"""
API Routes for Startup Studio Services

This module provides authenticated API endpoints for:
- Idea management and submission
- Pitch deck generation  
- Campaign creation and management
- System monitoring and health checks

All routes require authentication through the API Gateway.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from pipeline.api.gateway import gateway
from pipeline.models.idea import Idea
from pipeline.services.campaign_generator import CampaignGenerator
from pipeline.services.pitch_deck_generator import PitchDeckGenerator
from pipeline.storage.idea_repository import IdeaRepository
from pipeline.config.settings import get_settings

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["startup-studio"])

# Pydantic models for API requests/responses

class IdeaSubmissionRequest(BaseModel):
    """Request model for idea submission."""
    title: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=50, max_length=5000)
    category: str = Field(..., max_length=50)
    problem_statement: str = Field(..., min_length=20, max_length=2000)
    solution_description: str = Field(..., min_length=20, max_length=2000)
    target_market: str = Field(..., min_length=10, max_length=500)
    evidence_links: List[str] = Field(default=[])

class IdeaResponse(BaseModel):
    """Response model for idea data."""
    id: str
    title: str
    description: str
    category: str
    status: str
    created_at: datetime
    updated_at: datetime
    validation_score: Optional[float] = None

class PitchDeckRequest(BaseModel):
    """Request model for pitch deck generation."""
    idea_id: str
    template: Optional[str] = "default"
    include_financials: bool = True
    include_competition: bool = True

class PitchDeckResponse(BaseModel):
    """Response model for pitch deck generation."""
    deck_id: str
    idea_id: str
    deck_url: str
    generated_at: datetime
    status: str

class CampaignRequest(BaseModel):
    """Request model for campaign creation."""
    idea_id: str
    campaign_type: str  # "google_ads", "social_media", "email"
    budget: float
    duration_days: int = 30
    target_audience: Optional[str] = None

class CampaignResponse(BaseModel):
    """Response model for campaign creation."""
    campaign_id: str
    idea_id: str
    campaign_type: str
    status: str
    budget: float
    created_at: datetime

# Dependency to get current authenticated user
async def get_current_user(user: Dict[str, Any] = Depends(gateway.get_current_user)):
    """Get current authenticated user."""
    return user

# Initialize services
settings = get_settings()
idea_repository = IdeaRepository()
pitch_generator = PitchDeckGenerator()
campaign_generator = CampaignGenerator()

# Idea Management Routes

@router.post("/ideas", response_model=IdeaResponse, status_code=status.HTTP_201_CREATED)
async def submit_idea(
    idea_request: IdeaSubmissionRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Submit a new startup idea for validation and processing.
    
    Requires authentication. Idea will be validated and processed asynchronously.
    """
    try:
        # Create idea instance
        idea = Idea(
            title=idea_request.title,
            description=idea_request.description,
            category=idea_request.category,
            problem_statement=idea_request.problem_statement,
            solution_description=idea_request.solution_description,
            target_market=idea_request.target_market,
            evidence_links=idea_request.evidence_links
        )
        
        # Save idea
        saved_idea = await idea_repository.save_idea(idea)
        
        # Schedule background processing
        background_tasks.add_task(process_idea_async, saved_idea.id)
        
        logger.info(
            f"Idea submitted successfully",
            extra={
                "idea_id": saved_idea.id,
                "user_session": user.get("session_id"),
                "title": idea_request.title
            }
        )
        
        return IdeaResponse(
            id=saved_idea.id,
            title=saved_idea.title,
            description=saved_idea.description,
            category=saved_idea.category,
            status=saved_idea.status.value,
            created_at=saved_idea.created_at,
            updated_at=saved_idea.updated_at
        )
        
    except Exception as e:
        logger.error(f"Failed to submit idea: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit idea"
        )

@router.get("/ideas", response_model=List[IdeaResponse])
async def list_ideas(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    status_filter: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List submitted ideas with filtering and pagination.
    
    Requires authentication.
    """
    try:
        ideas = await idea_repository.list_ideas(
            limit=limit,
            offset=offset,
            category=category,
            status=status_filter
        )
        
        return [
            IdeaResponse(
                id=idea.id,
                title=idea.title,
                description=idea.description,
                category=idea.category,
                status=idea.status.value,
                created_at=idea.created_at,
                updated_at=idea.updated_at,
                validation_score=getattr(idea, 'validation_score', None)
            )
            for idea in ideas
        ]
        
    except Exception as e:
        logger.error(f"Failed to list ideas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ideas"
        )

@router.get("/ideas/{idea_id}", response_model=IdeaResponse)
async def get_idea(
    idea_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get details for a specific idea.
    
    Requires authentication.
    """
    try:
        idea = await idea_repository.get_idea(idea_id)
        if not idea:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Idea not found"
            )
        
        return IdeaResponse(
            id=idea.id,
            title=idea.title,
            description=idea.description,
            category=idea.category,
            status=idea.status.value,
            created_at=idea.created_at,
            updated_at=idea.updated_at,
            validation_score=getattr(idea, 'validation_score', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get idea {idea_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve idea"
        )

# Pitch Deck Generation Routes

@router.post("/pitch-decks", response_model=PitchDeckResponse)
async def generate_pitch_deck(
    pitch_request: PitchDeckRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Generate a pitch deck for an idea.
    
    Requires authentication. Deck generation is processed asynchronously.
    """
    try:
        # Verify idea exists
        idea = await idea_repository.get_idea(pitch_request.idea_id)
        if not idea:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Idea not found"
            )
        
        # Start deck generation
        deck_id = f"deck-{idea.id}-{int(datetime.utcnow().timestamp())}"
        
        # Schedule background generation
        background_tasks.add_task(
            generate_deck_async,
            deck_id,
            pitch_request.idea_id,
            pitch_request.template,
            pitch_request.include_financials,
            pitch_request.include_competition
        )
        
        logger.info(
            f"Pitch deck generation started",
            extra={
                "deck_id": deck_id,
                "idea_id": pitch_request.idea_id,
                "user_session": user.get("session_id")
            }
        )
        
        return PitchDeckResponse(
            deck_id=deck_id,
            idea_id=pitch_request.idea_id,
            deck_url=f"/api/v1/pitch-decks/{deck_id}",
            generated_at=datetime.utcnow(),
            status="generating"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start pitch deck generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate pitch deck"
        )

@router.get("/pitch-decks/{deck_id}")
async def get_pitch_deck(
    deck_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get pitch deck status and download link.
    
    Requires authentication.
    """
    try:
        # This would integrate with actual deck storage
        # For now, return status information
        return {
            "deck_id": deck_id,
            "status": "completed",
            "download_url": f"/api/v1/pitch-decks/{deck_id}/download",
            "generated_at": datetime.utcnow(),
            "format": "pdf"
        }
        
    except Exception as e:
        logger.error(f"Failed to get pitch deck {deck_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pitch deck"
        )

# Campaign Management Routes

@router.post("/campaigns", response_model=CampaignResponse)
async def create_campaign(
    campaign_request: CampaignRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a marketing campaign for an idea.
    
    Requires authentication. Campaign setup is processed asynchronously.
    """
    try:
        # Verify idea exists
        idea = await idea_repository.get_idea(campaign_request.idea_id)
        if not idea:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Idea not found"
            )
        
        # Validate budget
        if campaign_request.budget <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Budget must be positive"
            )
        
        # Create campaign
        campaign_id = f"campaign-{idea.id}-{int(datetime.utcnow().timestamp())}"
        
        # Schedule background campaign setup
        background_tasks.add_task(
            create_campaign_async,
            campaign_id,
            campaign_request.idea_id,
            campaign_request.campaign_type,
            campaign_request.budget,
            campaign_request.duration_days,
            campaign_request.target_audience
        )
        
        logger.info(
            f"Campaign creation started",
            extra={
                "campaign_id": campaign_id,
                "idea_id": campaign_request.idea_id,
                "campaign_type": campaign_request.campaign_type,
                "budget": campaign_request.budget,
                "user_session": user.get("session_id")
            }
        )
        
        return CampaignResponse(
            campaign_id=campaign_id,
            idea_id=campaign_request.idea_id,
            campaign_type=campaign_request.campaign_type,
            status="creating",
            budget=campaign_request.budget,
            created_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create campaign"
        )

@router.get("/campaigns/{campaign_id}")
async def get_campaign(
    campaign_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get campaign status and performance metrics.
    
    Requires authentication.
    """
    try:
        # This would integrate with actual campaign tracking
        # For now, return mock status information
        return {
            "campaign_id": campaign_id,
            "status": "active",
            "metrics": {
                "impressions": 10000,
                "clicks": 250,
                "conversions": 15,
                "spend": 45.67,
                "ctr": 2.5,
                "conversion_rate": 6.0
            },
            "updated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get campaign {campaign_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve campaign"
        )

# Background task functions

async def process_idea_async(idea_id: str):
    """Process idea asynchronously - validation, market research, etc."""
    try:
        logger.info(f"Starting async processing for idea {idea_id}")
        # This would trigger the full idea processing pipeline
        # For now, just log that processing started
        logger.info(f"Async processing completed for idea {idea_id}")
    except Exception as e:
        logger.error(f"Failed to process idea {idea_id}: {e}")

async def generate_deck_async(deck_id: str, idea_id: str, template: str, include_financials: bool, include_competition: bool):
    """Generate pitch deck asynchronously."""
    try:
        logger.info(f"Starting deck generation for {deck_id}")
        # This would trigger actual deck generation
        logger.info(f"Deck generation completed for {deck_id}")
    except Exception as e:
        logger.error(f"Failed to generate deck {deck_id}: {e}")

async def create_campaign_async(campaign_id: str, idea_id: str, campaign_type: str, budget: float, duration_days: int, target_audience: Optional[str]):
    """Create marketing campaign asynchronously.""" 
    try:
        logger.info(f"Starting campaign creation for {campaign_id}")
        # This would trigger actual campaign setup
        logger.info(f"Campaign creation completed for {campaign_id}")
    except Exception as e:
        logger.error(f"Failed to create campaign {campaign_id}: {e}")

# Add routes to gateway app
gateway.app.include_router(router)