# Dockerfile with same effect as command 
# docker run -p 9200:9200   -e "discovery.type=single-node"   -e "xpack.security.enabled=false"   -e "xpack.license.self_generated.type=trial"   docker.elastic.co/elasticsearch/elasticsearch:8.13.2
# Use the official Elasticsearch image from Docker Hub
FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.2

# Set environment variables
ENV discovery.type=single-node
ENV xpack.security.enabled=false
ENV xpack.license.self_generated.type=trial

# Expose the necessary port
EXPOSE 9200

# Default command to run Elasticsearch
CMD ["elasticsearch"]