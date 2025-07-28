# ADR-002: PostgreSQL with pgvector for Vector Storage and Similarity Search

## 1. Title

Selection of PostgreSQL + pgvector for Vector Storage and Similarity Search

## 2. Status

Accepted

## 3. Context

The Agentic Startup Studio requires high-performance vector similarity search for:
- Duplicate idea detection (semantic similarity)
- Evidence matching and citation finding
- Market research clustering and analysis
- Investor recommendation matching

Requirements:
- Sub-50ms similarity queries for 1M+ idea embeddings
- ACID compliance for transactional data integrity
- Scalable indexing for high-dimensional vectors (1536D OpenAI embeddings)
- Production-ready with backup, monitoring, and maintenance tooling
- Cost-effective operation within budget constraints

## 4. Decision

We have selected **PostgreSQL with pgvector extension** as our primary vector storage and similarity search solution with the following configuration:

### Technical Implementation
- **Database**: PostgreSQL 14+ with pgvector extension
- **Indexing**: HNSW (Hierarchical Navigable Small World) for <50ms queries
- **Vector Dimensions**: 1536D (OpenAI text-embedding-3-small)
- **Distance Metric**: Cosine similarity for semantic matching
- **Connection Pooling**: Asyncpg with connection pool management

### Performance Optimizations
- **Index Configuration**: `m=16, ef_construction=64` for optimal speed/accuracy balance
- **Query Optimization**: `ef_search=40` for production queries
- **Batch Processing**: Vectorized operations for bulk similarity searches
- **Caching Strategy**: Vector cache for frequently accessed embeddings

### Operational Features
- **Backup**: Automated PostgreSQL backups with point-in-time recovery
- **Monitoring**: Native PostgreSQL monitoring with custom vector metrics
- **Scaling**: Read replicas for query distribution
- **Maintenance**: Automated index optimization and statistics updates

## 5. Consequences

### Positive Consequences
- **Performance**: Achieves <50ms similarity queries with HNSW indexing
- **Reliability**: PostgreSQL's ACID compliance ensures data consistency
- **Ecosystem**: Rich tooling ecosystem for monitoring, backup, and management
- **Cost Efficiency**: Single database reduces operational complexity and costs
- **SQL Integration**: Familiar SQL interface for complex analytical queries
- **Vector + Relational**: Combines vector search with relational data in single system

### Negative Consequences
- **Memory Usage**: HNSW indexes require significant RAM for large datasets
- **Index Build Time**: Initial index construction can take hours for large datasets
- **Extension Dependency**: Relies on pgvector extension availability and updates
- **Scaling Limits**: Single-node limitations for extremely large vector datasets
- **Expertise Required**: Team needs PostgreSQL optimization expertise

## 6. Alternatives

### Dedicated Vector Databases
- **Pinecone**: 
  - **Rejected**: High cost, vendor lock-in, external dependency
- **Weaviate**: 
  - **Rejected**: Additional operational overhead, learning curve
- **Qdrant**: 
  - **Rejected**: Less mature ecosystem, separate database to maintain

### Elasticsearch with Vector Search
- **Rejected**: Higher operational complexity, less mature vector capabilities
- **Issues**: JVM resource requirements, complex cluster management

### Redis with Vector Extensions
- **Rejected**: In-memory limitations, less ACID compliance
- **Issues**: Data persistence concerns, memory cost scaling

### Embeddings in Application Memory
- **Rejected**: Scalability limitations, instance restart data loss
- **Issues**: No persistence, poor query performance at scale

## 7. References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Performance Tuning](https://www.postgresql.org/docs/current/performance-tips.html)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [Vector Database Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## 8. Date

2025-07-28

## 9. Authors

- Terragon Labs Engineering Team  
- Claude Code AI Assistant