#Get a table of the top 30 features from a random forest discrete classification model run in R
##written using randomForest 4.6-7

library(randomForest)
a = readRDS('discData.0.rf')

##Collect features from all three forests
v1 <- a$r[[1]]$importance
x1<-v1[order(-v1[, 3]), ]
names1<-rownames(x1[1:30,])

v2 <- a$r[[2]]$importance
x2<-v2[order(-v2[, 3]), ]
names2<-rownames(x2[1:30,])

v3 <- a$r[[3]]$importance
x3<-v3[order(-v3[, 3]), ]
names3<-rownames(x3[1:30,])


##make a list of all emements and get just the ones that occur more than once 
all<-c(names1,names2,names3)
n_occur <- data.frame(table(all))
n_occur[n_occur$Freq > 1,]
concen<-unique(all[all %in% n_occur$all[n_occur$Freq > 1]])

#get MDA for each feature from each forest
mda1<-x1[concen,"MeanDecreaseAccuracy"]
mda2<-x2[concen,"MeanDecreaseAccuracy"]
mda3<-x3[concen,"MeanDecreaseAccuracy"]
vals<-data.frame(mda1,mda2,mda3)
vals$avgMDA <- rowMeans(vals[,])

#Write it up as a text file that can be fed into plotting scripts
write.table(vals,file='disc_top30_wMDA.txt',sep='\t',eol='\n',row.names=TRUE,col.names=TRUE)

