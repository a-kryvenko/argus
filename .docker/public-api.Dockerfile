FROM golang:1.26-alpine AS build

WORKDIR /src/apps/api/api-go
COPY apps/api/api-go/go.mod apps/api/api-go/go.sum ./
COPY apps/prophet/api-go/go.mod /src/apps/prophet/api-go/go.mod
COPY packages/common/src/common/observationcatalog /src/packages/common/src/common/observationcatalog
RUN go mod download
COPY apps/api/api-go/*.go ./
COPY apps/prophet/api-go/publicforecast /src/apps/prophet/api-go/publicforecast
COPY apps/api/api-go/testdata ./testdata
COPY configs /src/configs
ENV CGO_ENABLED=0
RUN go test ./... && go build -trimpath -o /public-api .

FROM alpine:3.23
COPY --from=build /public-api /usr/local/bin/public-api
USER 65534:65534
EXPOSE 8000
CMD ["public-api"]
