#### preprocess the compas dataset

# which data is going to use?
# ProPublica COMPAS
data_compas = read.csv("~/Desktop/compas-scores-two-years.csv",header = F,stringsAsFactors = F,sep = ',')
compas = data_compas[1:7215,c(6,8,10:15,23,25,34,40,44,53)]
compas = compas[which(compas[,3]%in%c('African-American','Caucasian','Hispanic','Native American')),]
compas[,4] = as.numeric(compas[,4])
compas[which(compas[,3]%in%c('African-American','Native American')),3] = 'Black'
compas[which(compas[,3]%in%c('Caucasian','Hispanic')),3] = 'White'


# use statistical parity
# change everything into dummy
compas_data = compas
compas_data[,1] = 1*(compas[,1]=="Male")
compas_data[,3] = 1*(compas[,3]=="White")
compas_data[,9] = 1*(compas[,9]=="F")
compas_data = apply(compas_data,2,as.numeric)

compas_gender = compas[,1]
compas_race = compas[,3]
compas_data_imp = scale(compas_data[,-c(1,3)])

# save the data
save(compas_data, file = "~/Desktop/compas_data.RData")
# or
write.csv(compas_data,file = "~/Desktop/compas.csv")


