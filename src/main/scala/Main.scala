
// Step 1 — Import packages, load, parse, and explore the movie and rating dataset

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation._              //Rating
import scala.Tuple2
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._


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

        val (trainingData, testData) = (splits(0), splits(1))

        val numTraining = trainingData.count()

        val numTest = testData.count()

        println("Training: " + numTraining + " test: " + numTest)


 // Step 5 — Prepare the data for building the recommendation model using ALS

        
        val ratings_ML_DS = trainingData.map(row => Rating(row.userId,row.movieId,row.rating))

        val test_ML_DS = testData.map(row => Rating(row.userId,row.movieId,row.rating))


//  Step 6 — Build an ALS user product matrix


        val rank = 20

        val numIterations = 15

        val lambda = 0.10

        val alpha = 1.00 

        val block = -1

        val seed = 12345L

        val implicitPrefs = false

        val model = new ALS().setIterations(numIterations) .setBlocks(block).setAlpha(alpha)

        .setLambda(lambda)

        .setRank(rank) .setSeed(seed)

        .setImplicitPrefs(implicitPrefs)

        .run(ratings_ML_DS.rdd)


//      Step 7 — Making predictions


        // Making Predictions. Get the top 5 movie predictions for user 6

            println("Rating:(UserID, MovieID, Rating)")

            println("----------------------------------")

            val topRecsForUser = model.recommendProducts(6, 5) 
            for (rating <- topRecsForUser) { println(rating.toString()) } 
            println("----------------------------------")


 //     Step 8 — Evaluating the model

        // val rmseTest = computeRmse(model, test_ML_DS, true)

        // println("Test RMSE: = " + rmseTest) //Less is better


        sc.stop()


        }



    }