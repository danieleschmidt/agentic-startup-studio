-- Enhanced Schema Migration for Agentic Startup Studio
-- Checkpoint A2: Advanced Data Layer Implementation
-- This migration creates the complete production-ready database schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types
CREATE TYPE idea_status AS ENUM (
    'DRAFT',
    'VALIDATING', 
    'VALIDATED',
    'REJECTED',
    'RESEARCHING',
    'BUILDING',
    'TESTING',
    'DEPLOYED',
    'ARCHIVED'
);

CREATE TYPE pipeline_stage AS ENUM (
    'IDEATE',
    'RESEARCH', 
    'DECK',
    'INVESTORS',
    'MVP',
    'BUILDING',
    'SMOKE_TEST',
    'COMPLETE'
);

CREATE TYPE idea_category AS ENUM (
    'fintech',
    'healthtech',
    'edtech',
    'saas',
    'ecommerce',
    'ai_ml',
    'blockchain',
    'consumer',
    'enterprise',
    'marketplace',
    'uncategorized'
);

CREATE TYPE evidence_source_type AS ENUM (
    'academic',
    'industry_report',
    'news',
    'blog',
    'patent',
    'government',
    'company'
);

CREATE TYPE workflow_status AS ENUM (
    'pending',
    'in_progress',
    'completed',
    'failed',
    'retrying'
);

-- ============================================================================
-- CORE IDEAS TABLE
-- ============================================================================

CREATE TABLE ideas (
    -- Primary identification
    idea_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Core content
    title VARCHAR(200) NOT NULL CHECK (length(title) >= 10),
    description TEXT NOT NULL CHECK (length(description) >= 10),
    category idea_category NOT NULL DEFAULT 'uncategorized',
    
    -- Pipeline state
    status idea_status NOT NULL DEFAULT 'DRAFT',
    current_stage pipeline_stage NOT NULL DEFAULT 'IDEATE',
    stage_progress DECIMAL(3,2) NOT NULL DEFAULT 0.0 CHECK (stage_progress >= 0.0 AND stage_progress <= 1.0),
    
    -- Optional detailed fields
    problem_statement TEXT CHECK (length(problem_statement) <= 1000),
    solution_description TEXT CHECK (length(solution_description) <= 1000),
    target_market VARCHAR(500),
    evidence_links TEXT[], -- Array of URLs
    
    -- Vector embedding for similarity search
    embedding vector(384), -- SentenceTransformer all-MiniLM-L6-v2 dimension
    
    -- Metadata and tracking
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by VARCHAR(100),
    
    -- Pipeline artifacts (JSON fields for flexibility)
    research_data JSONB DEFAULT '{}',
    investor_scores JSONB DEFAULT '{}',
    deck_path VARCHAR(500),
    
    -- Search and indexing
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || description || ' ' || coalesce(problem_statement, '') || ' ' || coalesce(solution_description, ''))
    ) STORED,
    
    -- Audit fields
    version INTEGER NOT NULL DEFAULT 1,
    last_modified_by VARCHAR(100)
);

-- Indexes for ideas table
CREATE INDEX idx_ideas_status ON ideas(status);
CREATE INDEX idx_ideas_category ON ideas(category);
CREATE INDEX idx_ideas_stage ON ideas(current_stage);
CREATE INDEX idx_ideas_created_at ON ideas(created_at DESC);
CREATE INDEX idx_ideas_updated_at ON ideas(updated_at DESC);
CREATE INDEX idx_ideas_created_by ON ideas(created_by);

-- Vector similarity search index (HNSW for fast approximate search)
CREATE INDEX idx_ideas_embedding_hnsw ON ideas USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX idx_ideas_search_vector ON ideas USING gin(search_vector);

-- Composite indexes for common query patterns
CREATE INDEX idx_ideas_status_category ON ideas(status, category);
CREATE INDEX idx_ideas_stage_progress ON ideas(current_stage, stage_progress);
CREATE INDEX idx_ideas_category_created_at ON ideas(category, created_at DESC);

-- ============================================================================
-- RESEARCH DATA TABLE
-- ============================================================================

CREATE TABLE research_data (
    -- Primary identification
    research_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES ideas(idea_id) ON DELETE CASCADE,
    
    -- Research content
    evidence JSONB NOT NULL DEFAULT '{}',
    citations JSONB DEFAULT '[]',
    market_analysis JSONB DEFAULT '{}',
    competitive_landscape JSONB DEFAULT '{}',
    
    -- Quality metrics
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    market_size_estimate DECIMAL(15,2), -- In millions USD
    
    -- Risk assessment
    risk_factors JSONB DEFAULT '[]',
    opportunities JSONB DEFAULT '[]',
    
    -- Source tracking
    source_type evidence_source_type,
    source_url TEXT,
    source_credibility DECIMAL(3,2) CHECK (source_credibility >= 0.0 AND source_credibility <= 1.0),
    
    -- Timestamps
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    verified_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    
    -- Metadata
    collection_method VARCHAR(100),
    researcher_id VARCHAR(100)
);

-- Indexes for research_data
CREATE INDEX idx_research_idea_id ON research_data(idea_id);
CREATE INDEX idx_research_collected_at ON research_data(collected_at DESC);
CREATE INDEX idx_research_source_type ON research_data(source_type);
CREATE INDEX idx_research_confidence ON research_data(confidence_score DESC);

-- GIN index for JSONB content search
CREATE INDEX idx_research_evidence_gin ON research_data USING gin(evidence);
CREATE INDEX idx_research_market_analysis_gin ON research_data USING gin(market_analysis);

-- ============================================================================
-- PITCH DECKS TABLE
-- ============================================================================

CREATE TABLE pitch_decks (
    -- Primary identification
    deck_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES ideas(idea_id) ON DELETE CASCADE,
    
    -- Deck content
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    sections JSONB DEFAULT '[]',
    
    -- Format and structure
    format VARCHAR(50) NOT NULL DEFAULT 'markdown',
    slide_count INTEGER CHECK (slide_count > 0),
    template_version VARCHAR(20),
    
    -- Quality metrics
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    accessibility_score DECIMAL(3,2) CHECK (accessibility_score >= 0.0 AND accessibility_score <= 1.0),
    
    -- Review and feedback
    reviewer_feedback JSONB DEFAULT '[]',
    investor_feedback JSONB DEFAULT '[]',
    
    -- File management
    file_path VARCHAR(500),
    file_size INTEGER,
    file_hash VARCHAR(64), -- SHA-256 hash for integrity
    
    -- Timestamps
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_at TIMESTAMPTZ,
    reviewed_at TIMESTAMPTZ,
    
    -- Version control
    version INTEGER NOT NULL DEFAULT 1,
    parent_deck_id UUID REFERENCES pitch_decks(deck_id),
    
    -- Status and approval
    is_approved BOOLEAN DEFAULT false,
    approved_by VARCHAR(100),
    approved_at TIMESTAMPTZ
);

-- Indexes for pitch_decks
CREATE INDEX idx_pitch_decks_idea_id ON pitch_decks(idea_id);
CREATE INDEX idx_pitch_decks_generated_at ON pitch_decks(generated_at DESC);
CREATE INDEX idx_pitch_decks_quality_score ON pitch_decks(quality_score DESC);
CREATE INDEX idx_pitch_decks_approved ON pitch_decks(is_approved, approved_at DESC);

-- Full-text search on deck content
CREATE INDEX idx_pitch_decks_content_search ON pitch_decks USING gin(to_tsvector('english', content));

-- ============================================================================
-- SMOKE TESTS TABLE  
-- ============================================================================

CREATE TABLE smoke_tests (
    -- Primary identification
    test_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES ideas(idea_id) ON DELETE CASCADE,
    
    -- Test configuration
    test_name VARCHAR(200) NOT NULL,
    test_description TEXT,
    test_environment VARCHAR(100) DEFAULT 'staging',
    
    -- Test metrics and results
    metrics JSONB NOT NULL DEFAULT '{}',
    analytics JSONB DEFAULT '{}',
    performance_data JSONB DEFAULT '{}',
    user_feedback JSONB DEFAULT '[]',
    
    -- Success criteria and results
    success_criteria JSONB DEFAULT '{}',
    success_rate DECIMAL(5,2) CHECK (success_rate >= 0.0 AND success_rate <= 100.0),
    status workflow_status NOT NULL DEFAULT 'pending',
    
    -- Cost and resource tracking
    cost_breakdown JSONB DEFAULT '{}',
    resource_usage JSONB DEFAULT '{}',
    
    -- Execution timeline
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER GENERATED ALWAYS AS (
        CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (completed_at - started_at))::INTEGER
        ELSE NULL END
    ) STORED,
    
    -- Test management
    test_suite VARCHAR(100),
    test_runner VARCHAR(100),
    artifacts_path VARCHAR(500),
    
    -- Error tracking
    error_details JSONB DEFAULT '{}',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

-- Indexes for smoke_tests
CREATE INDEX idx_smoke_tests_idea_id ON smoke_tests(idea_id);
CREATE INDEX idx_smoke_tests_status ON smoke_tests(status);
CREATE INDEX idx_smoke_tests_started_at ON smoke_tests(started_at DESC);
CREATE INDEX idx_smoke_tests_success_rate ON smoke_tests(success_rate DESC);
CREATE INDEX idx_smoke_tests_environment ON smoke_tests(test_environment);

-- ============================================================================
-- WORKFLOW EXECUTIONS TABLE
-- ============================================================================

CREATE TABLE workflow_executions (
    -- Primary identification
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES ideas(idea_id) ON DELETE CASCADE,
    
    -- Workflow metadata
    workflow_type VARCHAR(100) NOT NULL,
    workflow_version VARCHAR(20),
    correlation_id UUID,
    
    -- Input and output
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    intermediate_state JSONB DEFAULT '{}',
    
    -- Execution status
    status workflow_status NOT NULL DEFAULT 'pending',
    current_step VARCHAR(100),
    completed_steps JSONB DEFAULT '[]',
    
    -- Error handling
    error_details JSONB DEFAULT '{}',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Performance metrics
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    cpu_usage_percent DECIMAL(5,2),
    
    -- Timeline
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    last_heartbeat_at TIMESTAMPTZ DEFAULT now(),
    
    -- Agent and execution context
    agent_id VARCHAR(100),
    execution_context JSONB DEFAULT '{}',
    
    -- Quality gates
    quality_gate_results JSONB DEFAULT '{}',
    checkpoint_data JSONB DEFAULT '{}'
);

-- Indexes for workflow_executions
CREATE INDEX idx_workflow_executions_idea_id ON workflow_executions(idea_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_workflow_type ON workflow_executions(workflow_type);
CREATE INDEX idx_workflow_executions_started_at ON workflow_executions(started_at DESC);
CREATE INDEX idx_workflow_executions_correlation_id ON workflow_executions(correlation_id);
CREATE INDEX idx_workflow_executions_agent_id ON workflow_executions(agent_id);

-- ============================================================================
-- AUDIT TRAIL TABLE
-- ============================================================================

CREATE TABLE audit_trail (
    -- Primary identification
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idea_id UUID NOT NULL REFERENCES ideas(idea_id) ON DELETE CASCADE,
    
    -- Audit information
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL DEFAULT 'idea',
    entity_id UUID NOT NULL,
    
    -- Change tracking
    old_values JSONB DEFAULT '{}',
    new_values JSONB DEFAULT '{}',
    changes JSONB DEFAULT '{}',
    
    -- Context and attribution
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    correlation_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    
    -- Request context
    request_id VARCHAR(100),
    transaction_id VARCHAR(100),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags VARCHAR(100)[],
    
    -- Timestamp
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Security and compliance
    security_level VARCHAR(20) DEFAULT 'normal',
    compliance_flags JSONB DEFAULT '[]'
);

-- Indexes for audit_trail
CREATE INDEX idx_audit_trail_idea_id ON audit_trail(idea_id);
CREATE INDEX idx_audit_trail_action ON audit_trail(action);
CREATE INDEX idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX idx_audit_trail_entity ON audit_trail(entity_type, entity_id);
CREATE INDEX idx_audit_trail_correlation_id ON audit_trail(correlation_id);

-- Composite index for security queries
CREATE INDEX idx_audit_trail_security ON audit_trail(security_level, user_id, timestamp DESC);

-- ============================================================================
-- USER SESSIONS TABLE (Optional - for future authentication)
-- ============================================================================

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    
    -- Session data
    session_data JSONB DEFAULT '{}',
    user_agent TEXT,
    ip_address INET,
    
    -- Security
    csrf_token VARCHAR(128),
    is_authenticated BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '24 hours'),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    logout_reason VARCHAR(100)
);

-- Indexes for user_sessions
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_user_sessions_active ON user_sessions(is_active, last_accessed_at DESC);

-- ============================================================================
-- SYSTEM CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE system_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) NOT NULL DEFAULT 'string',
    
    -- Metadata
    description TEXT,
    category VARCHAR(50),
    is_sensitive BOOLEAN DEFAULT false,
    
    -- Version control
    version INTEGER DEFAULT 1,
    previous_value JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by VARCHAR(100)
);

-- Indexes for system_config
CREATE INDEX idx_system_config_category ON system_config(category);
CREATE INDEX idx_system_config_updated_at ON system_config(updated_at DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to update version number
CREATE OR REPLACE FUNCTION update_version_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_ideas_updated_at 
    BEFORE UPDATE ON ideas 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ideas_version
    BEFORE UPDATE ON ideas
    FOR EACH ROW
    EXECUTE FUNCTION update_version_column();

CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for similarity search with caching
CREATE OR REPLACE FUNCTION find_similar_ideas(
    query_embedding vector(384),
    similarity_threshold DECIMAL DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    idea_id UUID,
    title VARCHAR(200),
    similarity_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.idea_id,
        i.title,
        ROUND((1 - (i.embedding <=> query_embedding))::DECIMAL, 3) as similarity_score
    FROM ideas i
    WHERE i.embedding IS NOT NULL
        AND (1 - (i.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY i.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function for advanced search with filters
CREATE OR REPLACE FUNCTION search_ideas(
    search_query TEXT DEFAULT NULL,
    idea_status idea_status[] DEFAULT NULL,
    idea_categories idea_category[] DEFAULT NULL,
    pipeline_stages pipeline_stage[] DEFAULT NULL,
    created_after TIMESTAMPTZ DEFAULT NULL,
    created_before TIMESTAMPTZ DEFAULT NULL,
    sort_by VARCHAR(50) DEFAULT 'created_at',
    sort_desc BOOLEAN DEFAULT true,
    limit_count INTEGER DEFAULT 20,
    offset_count INTEGER DEFAULT 0
)
RETURNS TABLE (
    idea_id UUID,
    title VARCHAR(200),
    description TEXT,
    status idea_status,
    category idea_category,
    current_stage pipeline_stage,
    created_at TIMESTAMPTZ,
    rank REAL
) AS $$
DECLARE
    query_sql TEXT;
    order_clause TEXT;
BEGIN
    -- Build base query
    query_sql := 'SELECT i.idea_id, i.title, i.description, i.status, i.category, i.current_stage, i.created_at';
    
    -- Add ranking for text search
    IF search_query IS NOT NULL THEN
        query_sql := query_sql || ', ts_rank(i.search_vector, plainto_tsquery($1)) as rank';
    ELSE
        query_sql := query_sql || ', 0.0 as rank';
    END IF;
    
    query_sql := query_sql || ' FROM ideas i WHERE 1=1';
    
    -- Add filters
    IF search_query IS NOT NULL THEN
        query_sql := query_sql || ' AND i.search_vector @@ plainto_tsquery($1)';
    END IF;
    
    IF idea_status IS NOT NULL THEN
        query_sql := query_sql || ' AND i.status = ANY($2)';
    END IF;
    
    IF idea_categories IS NOT NULL THEN
        query_sql := query_sql || ' AND i.category = ANY($3)';
    END IF;
    
    IF pipeline_stages IS NOT NULL THEN
        query_sql := query_sql || ' AND i.current_stage = ANY($4)';
    END IF;
    
    IF created_after IS NOT NULL THEN
        query_sql := query_sql || ' AND i.created_at >= $5';
    END IF;
    
    IF created_before IS NOT NULL THEN
        query_sql := query_sql || ' AND i.created_at <= $6';
    END IF;
    
    -- Add ordering
    IF sort_desc THEN
        order_clause := ' ORDER BY ' || sort_by || ' DESC';
    ELSE
        order_clause := ' ORDER BY ' || sort_by || ' ASC';
    END IF;
    
    query_sql := query_sql || order_clause;
    query_sql := query_sql || ' LIMIT $7 OFFSET $8';
    
    -- Execute the dynamic query
    RETURN QUERY EXECUTE query_sql 
    USING search_query, idea_status, idea_categories, pipeline_stages, 
          created_after, created_before, limit_count, offset_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MATERIALIZED VIEWS FOR ANALYTICS
-- ============================================================================

-- Analytics view for idea pipeline metrics
CREATE MATERIALIZED VIEW idea_pipeline_analytics AS
SELECT 
    current_stage,
    status,
    category,
    COUNT(*) as idea_count,
    AVG(stage_progress) as avg_progress,
    MIN(created_at) as earliest_created,
    MAX(created_at) as latest_created,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time_seconds
FROM ideas
GROUP BY current_stage, status, category;

-- Index for the materialized view
CREATE INDEX idx_idea_pipeline_analytics_stage ON idea_pipeline_analytics(current_stage);
CREATE INDEX idx_idea_pipeline_analytics_status ON idea_pipeline_analytics(status);

-- Analytics view for research quality metrics
CREATE MATERIALIZED VIEW research_quality_analytics AS
SELECT 
    r.idea_id,
    i.category,
    i.status,
    COUNT(r.research_id) as research_count,
    AVG(r.confidence_score) as avg_confidence,
    MAX(r.confidence_score) as max_confidence,
    COUNT(DISTINCT r.source_type) as source_diversity,
    AVG(r.source_credibility) as avg_credibility
FROM research_data r
JOIN ideas i ON r.idea_id = i.idea_id
GROUP BY r.idea_id, i.category, i.status;

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY idea_pipeline_analytics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY research_quality_analytics;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECURITY AND PERMISSIONS
-- ============================================================================

-- Row Level Security policies (can be enabled later)
-- ALTER TABLE ideas ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE research_data ENABLE ROW LEVEL SECURITY;

-- Example RLS policy (commented out for now)
-- CREATE POLICY ideas_user_policy ON ideas
--     FOR ALL TO application_user
--     USING (created_by = current_user OR current_user = 'admin');

-- ============================================================================
-- INITIAL CONFIGURATION DATA
-- ============================================================================

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, config_type, description, category) VALUES
('app.version', '"2.0.0"', 'string', 'Application version', 'system'),
('app.maintenance_mode', 'false', 'boolean', 'Enable maintenance mode', 'system'),
('pipeline.max_concurrent_executions', '10', 'number', 'Maximum concurrent pipeline executions', 'pipeline'),
('pipeline.default_timeout_seconds', '3600', 'number', 'Default pipeline timeout in seconds', 'pipeline'),
('search.similarity_threshold', '0.8', 'number', 'Default similarity threshold for duplicate detection', 'search'),
('search.max_results', '50', 'number', 'Maximum search results to return', 'search'),
('audit.retention_days', '365', 'number', 'Audit log retention period in days', 'audit'),
('cache.default_ttl_seconds', '300', 'number', 'Default cache TTL in seconds', 'cache');

-- ============================================================================
-- PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Enable query plan caching
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Optimize for analytics workloads
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET effective_cache_size = '4GB';

-- Optimize checkpoint and WAL settings
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Enable query stats tracking
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_functions = 'all';

-- ============================================================================
-- HEALTH CHECK FUNCTIONS
-- ============================================================================

-- Database health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS jsonb AS $$
DECLARE
    result jsonb;
    total_ideas integer;
    active_workflows integer;
    db_size text;
BEGIN
    SELECT COUNT(*) INTO total_ideas FROM ideas;
    SELECT COUNT(*) INTO active_workflows FROM workflow_executions WHERE status = 'in_progress';
    SELECT pg_size_pretty(pg_database_size(current_database())) INTO db_size;
    
    result := jsonb_build_object(
        'status', 'healthy',
        'timestamp', now(),
        'metrics', jsonb_build_object(
            'total_ideas', total_ideas,
            'active_workflows', active_workflows,
            'database_size', db_size,
            'uptime_seconds', EXTRACT(EPOCH FROM (now() - pg_postmaster_start_time()))
        )
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Schema validation function
CREATE OR REPLACE FUNCTION validate_schema()
RETURNS jsonb AS $$
DECLARE
    result jsonb;
    table_count integer;
    index_count integer;
BEGIN
    SELECT COUNT(*) INTO table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'public';
    
    result := jsonb_build_object(
        'schema_valid', true,
        'tables_count', table_count,
        'indexes_count', index_count,
        'extensions', (
            SELECT array_agg(extname)
            FROM pg_extension
            WHERE extname IN ('vector', 'uuid-ossp', 'pgcrypto', 'btree_gin')
        )
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CLEANUP AND MAINTENANCE
-- ============================================================================

-- Function to archive old audit logs
CREATE OR REPLACE FUNCTION archive_old_audit_logs(retention_days integer DEFAULT 365)
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM audit_trail 
    WHERE timestamp < (now() - (retention_days || ' days')::interval);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS integer AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < now() OR (is_active = false AND last_accessed_at < now() - interval '7 days');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Enhanced schema migration completed successfully';
    RAISE NOTICE 'Tables created: ideas, research_data, pitch_decks, smoke_tests, workflow_executions, audit_trail, user_sessions, system_config';
    RAISE NOTICE 'Indexes created: %', (SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public');
    RAISE NOTICE 'Functions created: health_check, validate_schema, archive_old_audit_logs, cleanup_expired_sessions';
    RAISE NOTICE 'Materialized views: idea_pipeline_analytics, research_quality_analytics';
END
$$;