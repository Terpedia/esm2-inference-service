#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-terpedia-esm2}"
REPOSITORY="${REPOSITORY:-terpedia}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short=12 HEAD)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/esm2-inference:${IMAGE_TAG}"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project "${PROJECT_ID}"
gcloud artifacts repositories describe "${REPOSITORY}" --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPOSITORY}" --repository-format docker \
    --location "${REGION}" --project "${PROJECT_ID}"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .

gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 8 \
  --memory 32Gi \
  --no-cpu-throttling \
  --min 0 \
  --max 3 \
  --concurrency 32 \
  --timeout 300 \
  --set-env-vars ESM2_DEVICE=cuda,ESM2_MAX_BATCH_SEQUENCES=16,ESM2_MAX_BATCH_TOKENS=8192,ESM2_BATCH_WAIT_MS=8 \
  --no-allow-unauthenticated

gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" \
  --format='value(status.url)'

