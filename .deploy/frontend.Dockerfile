# syntax=docker/dockerfile:1.7

ARG NODE_VERSION=22-slim

FROM node:${NODE_VERSION} AS base

RUN mkdir -p /var/www
WORKDIR /var/www

RUN corepack enable

FROM base AS deps
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/next.config.js apps/web/package.json apps/web/tsconfig.json ./apps/web/
COPY apps/web/app ./apps/web/app
COPY apps/web/fonts ./apps/web/fonts
COPY apps/web/public ./apps/web/public

RUN --mount=type=cache,id=pnpm,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile --filter web...

FROM base AS builder

RUN mkdir -p /var/www
WORKDIR /var/www

COPY --from=deps /var/www/package.json /var/www/pnpm-lock.yaml /var/www/pnpm-workspace.yaml ./
COPY --from=deps /var/www/node_modules ./node_modules
COPY --from=deps /var/www/apps ./apps

RUN pnpm --filter web build

FROM node:${NODE_VERSION} AS runner

RUN mkdir -p /var/www
WORKDIR /var/www

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

COPY --from=builder /var/www/apps/web/.next/standalone ./
COPY --from=builder /var/www/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder /var/www/apps/web/public ./apps/web/public

EXPOSE 3000

CMD ["node", "/var/www/apps/web/server.js"]