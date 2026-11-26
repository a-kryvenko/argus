FROM golang:1.26-alpine AS build

WORKDIR /src/apps/clio/api-go
COPY apps/clio/api-go/go.mod apps/clio/api-go/go.sum ./
COPY packages/common/src/common/observationcatalog /src/packages/common/src/common/observationcatalog
RUN go mod download
COPY apps/clio/api-go/*.go ./
ENV CGO_ENABLED=0
RUN go test ./... && go build -trimpath -o /clio-api .

FROM alpine:3.23
COPY --from=build /clio-api /usr/local/bin/clio-api
USER 65534:65534
EXPOSE 8000
CMD ["clio-api"]
