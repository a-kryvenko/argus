module github.com/a-kryvenko/argus/apps/api/api-go

go 1.25.0

require gopkg.in/yaml.v3 v3.0.1

require github.com/danielgtaylor/huma/v2 v2.38.0

require github.com/a-kryvenko/argus/apps/prophet/api-go v0.0.0

require github.com/a-kryvenko/argus/observationcatalog v0.0.0

replace github.com/a-kryvenko/argus/apps/prophet/api-go => ../../prophet/api-go

replace github.com/a-kryvenko/argus/observationcatalog => ../../../packages/common/src/common/observationcatalog
