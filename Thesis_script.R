
## change files to smaller format using only neccessary data
library(dplyr)
data <- read.csv("C:/Users/anna_/Desktop/OpenFace output/Ziggurat/Jury/DEiAIII1920_SM1_1Ziggurat_cam4_3FredericHuijnen_informedconsent_optionD.csv")
data_new <- data %>% 
  select(frame, face_id, timestamp, success, AU06_r, AU07_r, AU12_r, AU14_r)
View(data_new)
write.csv(data_new, "C:/Users/anna_/Desktop/Ziggurat_judge3.csv", row.names = FALSE)

#EDA with histogram
library(psych)

clean_data_1 <- read.csv("C:/Users/anna_/Desktop/Ziggurat_judge1.csv")
clean_data_2 <- read.csv("C:/Users/anna_/Desktop/Ziggurat_judge2goed.csv")
clean_data_3 <- read.csv("C:/Users/anna_/Desktop/Ziggurat_judge3goed.csv")
clean_data_pitch <- read.csv("C:/Users/anna_/Desktop/Ziggurat_pitch.csv")

summary(clean_data_1)
summary(clean_data_2)
summary(clean_data_3)
summary(clean_data_pitch)

describe(clean_data_1)
describe(clean_data_2)
describe(clean_data_3)
describe(clean_data_pitch)

hist(clean_data$AU06_r, freq = TRUE, main = "Histogram Ziggurat Judge 1", xlab = "AU06")
hist(clean_data$AU07_r, freq = TRUE, main = "Histogram Ziggurat Judge 1", xlab = "AU07" )
hist(clean_data$AU12_r, freq = TRUE, main = "Histogram Ziggurat Judge 1", xlab = "AU12")
hist(clean_data$AU14_r, freq = TRUE, main = "Histogram Ziggurat Judge 1", xlab = "AU14" )



## detect mimicry in csv files using Pearson correlation
library(dplyr)
detect_mimic <- read.csv2("C:/Users/anna_/Desktop/Ziggurat.csv")
pitch <- detect_mimic %>%
  filter(th2.pitch == "Ziggurat_judge1") # do this for judge1, judge2 and judge3
round(cor(pitch$AU06_r, pitch$max_AU06), digits = 2)
round(cor(pitch$AU07_r, pitch$max_AU07), digits = 2)
round(cor(pitch$AU12_r, pitch$max_AU12), digits = 2)
round(cor(pitch$AU14_r, pitch$max_AU14), digits = 2)

# check for significance
cor.test(pitch$AU06_r, pitch$max_AU06) 
cor.test(pitch$AU07_r, pitch$max_AU07)
cor.test(pitch$AU12_r, pitch$max_AU12)
cor.test(pitch$AU14_r, pitch$max_AU14)

## Spearman and kendall correlation between degree of mimicry and ranking
library(dplyr)
Mimicry_lijst <- read.csv2("C:/Users/anna_/Downloads/Mimicry_lijst.csv")
cleaned_mimic <- Mimicry_lijst %>% #clean the file by removing NA's and empty column
  subset(select = -X) %>%
  na.omit()

x = 1:75 # here the input x is the rows that need to be taken into account for the correlation
cleaned_mimic$AU06[x]
cor.test(cleaned_mimic$AU06[x], cleaned_mimic$Ranking[x], method = "spearman", use = "comple.obs")
cor.test(cleaned_mimic$AU07[x], cleaned_mimic$Ranking[x], method = "spearman", use = "comple.obs")
cor.test(cleaned_mimic$AU12[x], cleaned_mimic$Ranking[x], method = "spearman", use = "comple.obs")
cor.test(cleaned_mimic$AU14[x], cleaned_mimic$Ranking[x], method = "spearman", use = "comple.obs")
cor.test(cleaned_mimic$degree[x], cleaned_mimic$Ranking[x], method = "spearman", use = "comple.obs")

cor.test(cleaned_mimic$AU06[x], cleaned_mimic$Ranking[x], method = "kendall", use = "comple.obs")
cor.test(cleaned_mimic$AU07[x], cleaned_mimic$Ranking[x], method = "kendall", use = "comple.obs")
cor.test(cleaned_mimic$AU12[x], cleaned_mimic$Ranking[x], method = "kendall", use = "comple.obs")
cor.test(cleaned_mimic$AU14[x], cleaned_mimic$Ranking[x], method = "kendall", use = "comple.obs")
cor.test(cleaned_mimic$degree[x], cleaned_mimic$Ranking[x], method = "kendall", use = "comple.obs")


