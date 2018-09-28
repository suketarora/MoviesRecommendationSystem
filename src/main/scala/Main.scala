
// Step 1 — Import packages, load, parse, and explore the movie and rating dataset

import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.mllib.recommendation.Rating


    object Main extends App  {

    case class Ratings(userId:Int,movieId:Int,rating:Double,timestamp:Long) extends Serializable{}
    case class Movie  (movieId:Int,title:String,genres:String) extends Serializable {}



        override def main(arg: Array[String]): Unit = {


        var sparkConf = new SparkConf().setMaster("local").setAppName("MoviesRecommendationSystem")
        var sc = new SparkContext(sparkConf)
        val spark = SparkSession.builder().appName("test").master("local[2]").getOrCreate()


        sc.setLogLevel("ERROR")

         import spark.implicits._



        // # File location and type
        val file_location1 = "file:///home/suket/case_studies/MoviesRecomdationSystem/src/main/resources/movies.csv"
        val file_location2 = "file:///home/suket/case_studies/MoviesRecomdationSystem/src/main/resources/small_rating.csv"
        val file_type = "csv"

        // # CSV options
        val infer_schema = "false"
        val first_row_is_header = "true"
        val delimiter = ","



        val customSchemaForRating = StructType(Array(
                                                         StructField("userId", IntegerType, true),
                                                        StructField("movieId", IntegerType, true),
                                                        StructField("rating", DoubleType, true),
                                                        StructField("timestamp", LongType, true)
                                                    )
                                                )


        val customSchemaForMovie = StructType(Array(
                                                         StructField("movieId", IntegerType, true),
                                                        StructField("title", StringType, true),
                                                        StructField("genres", StringType, true)
                                                       
                                                    )
                                                ) 

         // # The applied options are for CSV files. For other file types, these will be ignored.
        val moviesDS = spark.read.format(file_type) 
        .option("inferSchema", infer_schema) 
        .option("header", first_row_is_header) 
        .option("sep", delimiter) 
        .schema(customSchemaForMovie)
        .load(file_location1).as[Movie]  


        // # The applied options are for CSV files. For other file types, these will be ignored.
        val ratingsDS = spark.read.format(file_type) 
        .option("inferSchema", infer_schema) 
        .option("header", first_row_is_header) 
        .option("sep", delimiter) 
        .schema(customSchemaForRating)
        .load(file_location2).as[Ratings]  


        // Step 2 — Register both DataFrames as temp tables to make querying easier

        ratingsDS.createOrReplaceTempView("ratings")

        moviesDS.createOrReplaceTempView("movies")

        val numRatings = ratingsDS.count()

        val numUsers = ratingsDS.select($"userId").distinct().count()

        val numMovies = ratingsDS.select($"movieId").distinct().count() 

        println()
        println()

        println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")


        // Step 3 — Explore and query for related statistics


        // Get the max, min ratings along with the count of users who have rated a movie.

        println()
        println("The max, min ratings along with the count of users who have rated a movie.")
        println()

        val results = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "

        + "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"

        + "min(ratings.rating) as minr,count(distinct userId) as cntu "

        + "FROM ratings group by ratings.movieId) movierates "

        + "join movies on movierates.movieId=movies.movieId "

        + "order by movierates.cntu desc")


        results.show


//      Top 10 active users and how many times they rated a movie

        val mostActiveUsersSchema = ratingsDS.groupBy("userId").count.orderBy($"count".desc).limit(10)
        mostActiveUsersSchema.show
        println()
        println("Top 10 active users and how many times they rated a movie")
        println()
        results.show()

// Let’s have a look at a particular user and find the movies that, say user, 6 rated higher than 4:

         println()
        println("Let’s have a look at a particular user and find the movies that, say user, 6 rated higher than 4 ")
        println()
        val results2 = moviesDS.join(ratingsDS,ratingsDS.col("movieId") === moviesDS.col("movieId")).filter(($"userId" === 6) && ($"rating" > 4))
         // moviesDS.join(ratingsDS,ratingsDS.col("movieId") === moviesDS.col("movieId")).where($"userId" === 6).where($"rating" > 4).show
         results2.show


//  Step 4 — Prepare training and test rating data and check the counts

        val splits = ratingsDS.randomSplit(Array(0.75, 0.25), seed = 12345L)

        val (training, test) = (splits(0), splits(1))

        val numTraining = training.count()

        val numTest = test.count()

        println("Training: " + numTraining + " test: " + numTest)


 // Step 5 — Prepare the data for building the recommendation model using ALS

        
        val ratings_ML_DS = training.map(row => Rating(row.userId,row.movieId,row.rating))

        val test_ML_DS = test.map(row => Rating(row.userId,row.movieId,row.rating))


//  Step 6 — Build an ALS user product matrix


     // Build the recommendation model using ALS on the training data
        val als = new ALS()
          .setMaxIter(5)
          .setRegParam(0.01)
          .setUserCol("userId")
          .setItemCol("movieId")
          .setRatingCol("rating")
        val model = als.fit(training)


 //     Step 7 — Evaluating the model
        
// Evaluate the model by computing the RMSE on the test data
// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        model.setColdStartStrategy("drop")
        val predictions = model.transform(test)
        val evaluator = new RegressionEvaluator()
          .setMetricName("rmse")
          .setLabelCol("rating")
          .setPredictionCol("prediction")
        val rmse = evaluator.evaluate(predictions)
        println(s"Root-mean-square error = $rmse")// Less is better


//      Step 8 — Making predictions



            // Generate top 10 movie recommendations for each user
            val userRecs = model.recommendForAllUsers(10)
            // Generate top 10 user recommendations for each movie
            val movieRecs = model.recommendForAllItems(10)

            // Generate top 10 movie recommendations for a specified set of users  (9)
            val users = ratingsDS.select(als.getUserCol).distinct().where($"userId" === 9)
          
            val userSubsetRecs = model.recommendForUserSubset(users, 5).collect

// userSubsetRecs: Array[org.apache.spark.sql.Row] = Array([9,WrappedArray([4148,4.9978666], [2959,4.994247], [1997,4.987907], [3798,4.963723], [1923,3.9982934])])


            // Generate top 10 user recommendations for a specified set of movies   (Any two users) 
            val movies = ratingsDS.select(als.getItemCol).distinct().limit(2)
            val movieSubSetRecs = model.recommendForItemSubset(movies, 5).collect
 // movieSubSetRecs: Array[org.apache.spark.sql.Row] = Array([1580,WrappedArray([7,2.9961877], [10,1.1922325], [5,0.9261053], [8,0.89429384], [6,0.72400767])], [3997,WrappedArray([1,3.497674], [2,2.0534189], [4,1.6735281], [3,1.1184243], [5,0.9664554])])



        sc.stop()


        }



    }