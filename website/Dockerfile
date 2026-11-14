# DistLLM website — static site, no build step.
#
# The pages ship as authored: every page references its own css/ and js/ paths
# and there is no bundling, minification or asset hashing. GitHub Pages serves
# the website/ directory verbatim, and this image does the same, so what you
# test locally is byte-identical to what is deployed.
#
# .dockerignore excludes node_modules, tests/, package files and this
# Dockerfile itself, so the COPY below is already just the served site.
FROM nginx:alpine

COPY nginx-production.conf /etc/nginx/conf.d/default.conf
COPY . /usr/share/nginx/html/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
