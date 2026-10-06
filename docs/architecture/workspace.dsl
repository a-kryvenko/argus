workspace "Argus Sunwatch" "Runtime boundaries and service ownership" {
    model {
        user = person "User" "Explores space weather and forecasts"
        client = softwareSystem "External API client" "Consumes public contracts"
        providers = softwareSystem "Observation providers" "SWPC, OMNI, JSOC, GONG and other sources"
        argus = softwareSystem "Argus Sunwatch" "Collects observations, publishes forecasts and assesses impacts" {
            web = container "Web" "Observation and forecast UI" "Next.js"
            api = container "Public API" "Public contracts, authentication and usage statistics" "Python / HTTP"
            clioHttp = container "Clio HTTP" "Stored observations and checksum-verified original files" "Python / HTTP"
            clioWorker = container "Clio worker" "Collection, backfill, aggregation and normalization" "Python / scheduler"
            prophetHttp = container "Prophet HTTP" "Stored releases, status and verification summaries" "Python / HTTP"
            prophetWorker = container "Prophet worker" "Input snapshots, product calculations, publication and verification" "Python / scheduler"
            intelligenceHttp = container "Intelligence HTTP" "On-demand LEO drag assessment using intelligence-core" "Python / HTTP"
            intelligenceWorker = container "Intelligence worker" "Release integration stub; does not calculate drag assessments" "Python / scheduler"
            apiDb = container "API database" "Sessions and usage statistics" "PostgreSQL / argus_api" "Database"
            clioDb = container "Clio database" "Measurements, aggregates, normalized observations and archive receipts" "PostgreSQL / clio" "Database"
            prophetDb = container "Prophet database" "Snapshots, forecasts, releases and verification records" "PostgreSQL / argus_prophet" "Database"
            intelligenceDb = container "Intelligence database" "Worker attempts and integration stub results" "PostgreSQL / argus_intelligence" "Database"
            archive = container "Original file archive" "AIA, HMI and new GONG originals; owned by Clio" "Shared disk" "Storage"
            models = container "Model artifacts" "Configured inference bundles; deployed explicitly" "Files" "Storage"
            metrics = container "Historical model metrics" "Static evaluation outputs; separate from operational verification" "Files" "Storage"
        }
        user -> web "Explores observations and forecasts" "HTTPS"
        client -> api "Reads public contracts" "HTTPS"
        web -> api "Reads observations, forecasts and assessments" "HTTP"
        api -> clioHttp "Reads observations" "HTTP"
        api -> prophetHttp "Reads forecasts and verification summaries" "HTTP"
        api -> intelligenceHttp "Requests drag assessment" "HTTP"
        api -> apiDb "Reads and writes sessions and usage" "SQL"
        api -> metrics "Reads historical evaluation metrics"
        clioWorker -> providers "Fetches observations and original files" "HTTP"
        clioWorker -> clioDb "Writes and prepares observations" "SQL"
        clioWorker -> archive "Writes original files"
        clioHttp -> clioDb "Reads stored observations and receipts" "SQL"
        clioHttp -> archive "Reads catalog and serves original files"
        prophetWorker -> clioHttp "Reads observations and downloads originals" "HTTP"
        prophetWorker -> models "Loads configured model bundles"
        prophetWorker -> prophetDb "Stores snapshots, releases and verification" "SQL"
        prophetHttp -> prophetDb "Reads stored releases and verification" "SQL"
        intelligenceHttp -> prophetHttp "Reads density release" "HTTP"
        intelligenceWorker -> prophetHttp "Polls solar-wind-speed releases" "HTTP"
        intelligenceWorker -> intelligenceDb "Writes attempts and stub results" "SQL"
    }
    views {
        systemContext argus "context" {
            include *
            autoLayout lr
        }
        container argus "containers" {
            include *
            autoLayout tb
        }
        styles {
            element "Element" {
                color #172554
                background #eff6ff
            }
            element "Person" {
                shape Person
            }
            element "Database" {
                shape Cylinder
                background #dbeafe
            }
            element "Storage" {
                shape Folder
                background #fef3c7
            }
        }
    }
}
