suppressMessages(library(geojsonio))
suppressMessages(library(leaflet))
suppressMessages(library(dplyr))
suppressMessages(library(data.table))
suppressMessages(library(shp2graph))
suppressMessages(library(sf))
suppressMessages(library(sp))

df <- fread('../geo_routing/2020-08-04.csv') %>% as.data.table

pal_temp34 <- colorQuantile(palette = 'YlOrRd', domain = df$temp34, n = 9)
qpal_colors <- unique(pal_temp34(sort(df$temp34))) # hex codes
df$temp34_clr <- pal_temp34(df$temp34)

col_legend <- seq(df$temp34 %>% min(), df$temp34 %>% max(), length.out = 9)
col_legend_cols <- pal_temp34(col_legend)

df_sf <- st_as_sf(df, coords = c("lon", "lat"), crs = 4326)

df$lat1 <- df$lat-0.5
df$lat2 <- df$lat+0.5
df$lon1 <- df$lon-0.5
df$lon2 <- df$lon+0.5

leaflet(options = leafletOptions(preferCanvas = TRUE)) %>%
  addTiles(options = providerTileOptions(
    updateWhenZooming = FALSE,      # map won't update tiles until zoom is done
    updateWhenIdle = TRUE           # map won't load new tiles when panning
  )) %>%
  addMapPane("circles", zIndex = 420) %>%
  addCircles(data = df_sf,
             color = ~temp34_clr, #Update to global/local pen/rev color
             radius = 3,
             fillOpacity = 0.4,
             popup = paste("arpu: ", df$avg_arpu,
                           "penetration: ", df$avg_penetration),
             highlightOptions = highlightOptions(color = "silver", weight = 5),
             options = pathOptions(pane = "circles")) %>%
  addMapPane("rectangles", zIndex = 410) %>%
  addRectangles(lng1=df$lon1,
                lat1=df$lat1,
                lng2=df$lon2,
                lat2=df$lat2,
                color = '#808080', #df$temp34_clr,
                fillColor = df$temp34_clr,
                weight = 0.5,
                fillOpacity = 0.5,
                highlightOptions = highlightOptions(color = "silver", weight = 5),
                options = pathOptions(pane = "rectangles")) %>%
  addLegend(position = c("topright"),
            colors = pal_temp34(col_legend),
            labels = round(col_legend,0),
            title = "Temp (degC)")

             