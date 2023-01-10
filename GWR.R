library(rgeos)
library(sp)
library(raster)

#read raster
cyr1 <- raster("C:\\Users\\zexir\\Desktop\\DSM test\\elevation.tiff")
cyr2 <- raster("C:\\Users\\zexir\\Desktop\\DSM test\\radK.tiff")
cyr3 <- raster("C:\\Users\\zexir\\Desktop\\DSM test\\NDVI.tif")
cyr4 <- raster("C:\\Users\\zexir\\Desktop\\DSM test\\slope\\slope.tif")
cyr5 <- raster("C:\\Users\\zexir\\Desktop\\DSM test\\aspect\\aspect1.tif")
cyrkp <- rasterToPoints(cyr1)

predictdatacykrcp26 <- data.frame(x = cyrkp[,1], 
                                  y = cyrkp[,2],
                                  ele = extract(raster("C:\\Users\\zexir\\Desktop\\DSM test\\elevation.tiff"),
                                                cyrkp[, c(1,2)]),
                                  radK = extract(raster("C:\\Users\\zexir\\Desktop\\DSM test\\radK.tiff"),
                                                 cyrkp[, c(1,2)]),
                                  ndvi = extract(raster("C:\\Users\\zexir\\Desktop\\DSM test\\NDVI.tif"),
                                                 cyrkp[, c(1,2)]),
                                  slope = extract(raster("C:\\Users\\zexir\\Desktop\\DSM test\\slope\\slope.tif"),
                                                  cyrkp[, c(1,2)]),
                                  aspect = extract(raster("C:\\Users\\zexir\\Desktop\\DSM test\\aspect\\aspect1.tif"),
                                                   cyrkp[, c(1,2)])
)

predictdatacykrcp26 <- na.omit(predictdatacykrcp26)
predictdatacykrcp26 <- predictdatacykrcp26[which(predictdatacykrcp26$ndvi!=-999),]

coordinates(predictdatacykrcp26 ) <- ~x+y
crs(cyr1)
cyr1
crs(predictdatacykrcp26) <- "+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs"

write.table(predictdatacykrcp26,file = "C:\\Users\\zexir\\Desktop\\DSM test\\raster\\nonsamplev2.txt")






library(GWmodel)
library(sf)
library(raster)
library(rgdal)
library(R)

fitusedatasp <- read.csv('C:\\Users\\zexir\\Desktop\\DSM test\\sample_soc_v1\\sample_soc_v7.csv')
coordinates(fitusedatasp) <- ~east+north
crs(cyr1)
cyr1
crs(fitusedatasp) <- "+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs"

crs(fitusedatasp)
crs(predictdatacykrcp26)

#fitusedatasp26 <- as.data.frame(fitusedatasp)

distancegwr <- bw.gwr(SOC_0_5_cm~ ndvi, fitusedatasp,approach = 'AICc',kernel='bisquare')
gwrcyk <- gwr.basic(SOC_0_5_cm ~ ndvi, fitusedatasp, bw = distancegwr,kernel='bisquare')
gwrcyk

#dm1=gw.dist()

#gwr output
gwrresult <- gwrcyk$SDF
#gwr.write.shp(gwrresult,fn="gwrresults")
writeOGR(gwrresult, "C:\\Users\\zexir\\Desktop\\DSM test\\gwr1.shp",
         "gwrresult", driver = "ESRI Shapefile")

deleteDataset(fitusedatasp26)

# prediction function

rcp26result  <- gwr.predict(SOC_0_5_cm~ ndvi, data = fitusedatasp, 
                           predictdata = predictdatacykrcp26, bw = distancegwr, kernel = "gaussian")


rcp26result <- gwr.predict(SOC_0_5_cm~ twi+ndvi, fitusedatasp,predictdatacykrcp26, bw = 20366.83, kernel = "gaussian",dMat1=Null)



rcp26result$SDF

help(gwr.predict())


spplot(rcp26result$SDF, zcol = "prediction")
