# Multi-stage build for DistLLM website
# Stage 1: Build with Vite
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY . .

# Build for production
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine AS production

# Copy custom nginx config (production variant with HSTS, rate limiting, security headers)
COPY nginx-production.conf /etc/nginx/conf.d/default.conf

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy static files that aren't processed by Vite
COPY --from=builder /app/robots.txt /usr/share/nginx/html/
COPY --from=builder /app/sitemap.xml /usr/share/nginx/html/
COPY --from=builder /app/feed.xml /usr/share/nginx/html/
COPY --from=builder /app/opensearch.xml /usr/share/nginx/html/
COPY --from=builder /app/favicon.svg /usr/share/nginx/html/
COPY --from=builder /app/manifest.json /usr/share/nginx/html/
COPY --from=builder /app/CNAME /usr/share/nginx/html/
COPY --from=builder /app/.htaccess /usr/share/nginx/html/

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
