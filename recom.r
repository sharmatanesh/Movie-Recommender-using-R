
#Load the ratings data from MovieLens 100k files.
ratings <- read.csv("C:/Users/user/Desktop/ml-latest-small/ratings.csv")

#install recommender lab package for building the recommendations.
#install.packages("recommenderlab")
library("recommenderlab")

#Create training and test data.
dt = sort(sample(nrow(ratings), nrow(ratings)*.7))
#train<-ratings[dt,]
#test<-ratings[-dt,]
rating_train <- ratings[dt,]

rating_test <- ratings[-dt,]

#remove the timestamp as it may not be required for this project
rating_train <- rating_train[,-c(4)]
names(rating_train)

rating_test <-rating_test[,-c(4)]
names(rating_test)
#[1] "userId" "movieId" "rating"

rating_test <-rating_test[,-c(3)]
names(rating_test)
#[1] "userId" "movieId"
library("reshape2")
#Cast the rating data into matrix form where row represensts users and columns items
cast_train_data<-acast( rating_train , rating_train$userId ~ rating_train$movieId )
class(cast_train_data)
rating_mat <- as.matrix(cast_train_data)

#Convert the matrix into a rating matrix form where each cell stores the ratings for user - movie combination
rating_real_mat <- as(rating_mat,"realRatingMatrix")

head(as(rating_real_mat,"list"),1)
head(as(rating_real_mat,"data.frame"),1)

#normalize ratings to view them in standard normal form
rating_norm <-normalize(rating_real_mat)

#plot a density based visual for users vs items matrix showing how users have rated items.
image(rating_real_mat,main="Raw Ratings")
image(rating_norm,main="Normalized Ratings")


#analyize the ratings data
library("ggplot2")
qplot(getRatings(rating_real_mat),binwidth=1,col="green",main="Histogram of Ratings",xlab="Ratings")
hist(getRatings(rating_real_mat),binwidth=1,col="green",breaks=15,main="Histogram of Ratings",xlab="Ratings");


qplot(getRatings(rating_norm),binwidth=1,main="Histogram of Normalized Ratings",xlab="Ratings")

qplot(rowCounts(rating_real_mat),binwidth=10,main="Movie Average Ratings",xlab="# of users",ylab = "# of movie ratings")
qplot(colMeans(rating_real_mat),binwidth=.1,main="Mean Ratings",xlab="Ratings",ylab = "# of movies")

rating_binary <- binarize(rating_real_mat,minRating=1)
head(as(rating_binary,"matrix"),1)

#create a recommender model using User based Collaborative Filtering method
#normalize the scores/ratings and use cosine similarity to group similar movies together
recommender <- Recommender(rating_real_mat[1:nrow(rating_real_mat)],method="UBCF",
                           param=list(normalize="Z-score",method="Cosine"
                                      ,nn=5,minRating=1))

print(recommender)
#Recommender of type 'UBCF' for 'realRatingMatrix'
#learned using 671 users.

names(getModel(recommender))
#[1] "description" "data" "method" "nn" "sample"
#[6] "normalize" "verbose"
#recommender
#Recommender of type 'UBCF' for 'realRatingMatrix'
#learned using 671 users.
getModel(recommender)$nn

#Use the recommender model to create a predictions for the ratings and top 10 movies
recommed <-predict(recommender,rating_real_mat[1:nrow(rating_real_mat)],type="ratings")
recommed_topN <-predict(recommender,rating_real_mat[1:nrow(rating_real_mat)],type="topNList",n=10)

head(as(recommed_topN,"list"),1)

as(recommed,"list")
as(recommed,"matrix")[,1:2]
as(recommed,"matrix")[1,2]

as.integer(as(recommed,"matrix")[5,10])

recommend_list <-as(recommed,"list")
user1 <- as.data.frame(recommend_list[[1]])
attributes(user1)
class(user1)
user1$uid <-row.names(user1)


# Function that calculates the ratings using the test data and the recommendation model
recommend_ratings<- function(rating_test_data,rec_list)
{
  for ( user in 1:length(rating_test_data[,2]))
  {
    # Read userid and movieid from columns 2 and 3 of test data
    userid <- rating_test_data[user,1]
    movieid<-rating_test_data[user,2]
    
    # Get as list & then convert to data frame all recommendations for user: userid
    u1<-as.data.frame(rec_list[[userid]])
    
    # Create a (second column) column-id in the data-frame u1 and populate it with row-names
    # Remember (or check) that rownames of u1 contain are by movie-ids
    # We use row.names() function
    u1$id<-row.names(u1)
    
    # Now access movie ratings in column 1 of u1
    x= u1[u1$id==movieid,1]
    
    ratings<-rep(0,length(rating_test_data[,2]));
    # print(u)
    # print(length(x))
    # If no ratings were found, assign 0. You could also
    # assign user-average
    if (length(x)==0)
    {
      ratings[user] <- 0
    }
    else
    {
      ratings[user] <-x
    }
    
  }
  return(ratings);
}

length(ratings)
tx<-merge(test[,1],round(ratings))

write.table(tx,file="submitfile.csv",row.names=FALSE,col.names=FALSE,sep=',')

recommend_top_list <-as(recommed_topN,"list")

# Function that returns the top 10 movies for a particular user
recommend_movies <- function(rec_list,movies,user_id)
{
  rec_movie_ids <- rec_list[[user_id]];
  movie_names<- movies[rec_movie_ids,];
  
  return(movie_names);
}

#Function Call
recommend_movies(recommend_top_list,movies,2)

############################Evaluation of Recommender Algorithms#################
#Build scheme to be used in evaulation process
schemes<-evaluationScheme(rating_real_mat,method="split",train=.8,k=1,given=10,goodRating=4)

schemes
#Evaluation scheme with 10 items given
#Method: 'split' with 1 run(s).
#Training set proportion: 0.800
#Good ratings: >=4.000000
#Data set: 671 x 9066 rating matrix of class 'realRatingMatrix' with 100004 ratings.


algorithms <- list(
  "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
  "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score")),
  "item-based CF" = list(name="IBCF2", param=list(normalize = "Z-score")),
  "SVD approximation"=list(name="SVD",param=list(k=50)));

results <- evaluate(schemes, algorithms, n=c(1, 3, 5, 10, 15, 20))

plot(results,annotate=1:3,legend="topleft")
plot(results,"prec/rec",annotate=3)

#############################SVD##########################
#install.packages("lsa")
#library("lsa")

#rating_ibf <-rating_mat[,-1]
#rating_ibf[is.na(rating_ibf)]=0

fsvd <- funkSVD(rating_real_mat,k=2,verbose = TRUE)
test_svd <- tcrossprod(fsvd$U,fsvd$V)
sqrt(MSE(rating_mat,test_svd))
#1.21024
