FROM golang:1.26-alpine AS build

WORKDIR /src
COPY apps/prophet/api-go/go.mod apps/prophet/api-go/go.sum ./
RUN go mod download
COPY apps/prophet/api-go/*.go ./
COPY apps/prophet/api-go/publicforecast ./publicforecast
ENV CGO_ENABLED=0
RUN go test ./... && go build -trimpath -o /prophet-api .

FROM alpine:3.23
COPY --from=build /prophet-api /usr/local/bin/prophet-api
EXPOSE 8000
CMD ["prophet-api"]
