using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace MLExperiments
{
    public class LegoSetsData
    {
        [Column("0")]
        public float ID;

        [Column("1")]
        public string Theme;

        [Column("2")]
        public float Year;

        [Column("3")]
        public float Licensed;

        [Column("4")]
        public float Pieces;

        [Column("5")]
        public float Minifigs;

        [Column("6")]
        public float Price;
    }

    public class LegoSetsPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice;
    }

    public class LegoSetsExperiment
    {
        static PredictionModel<LegoSetsData, LegoSetsPrediction> model;
        static string modelPath = "Data/lego_model.zip";
        static string modelStatsPath = "Data/lego_modelstats.txt";
        static string dataPath = "Data/lego-sets-training.txt";
        static string testDataPath = "Data/lego-sets-test.txt";

        public static void Execute()
        {
            Console.WriteLine("Executing Lego Sets Experiment");
            Console.WriteLine("Creating new model");
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<LegoSetsData>(dataPath, useHeader: true, separator: ","));
            pipeline.Add(new ColumnCopier(("Price", "Label")));

            pipeline.Add(new CategoricalOneHotVectorizer("Theme"));

            var features = new string[] { "Pieces", "Minifigs"};
            pipeline.Add(new ColumnConcatenator("Features", features));

            var algorithm = new FastTreeRegressor { NumLeaves = 6, NumTrees = 6, MinDocumentsInLeafs = 4};
            pipeline.Add(algorithm);

            model = pipeline.Train<LegoSetsData, LegoSetsPrediction>();

            var testData = new TextLoader<LegoSetsData>(testDataPath, useHeader: true, separator: ",");
            //pipeline.Add(new ColumnCopier(("Price", "Label")));
            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"L1: {metrics.L1}");
            Console.WriteLine($"L2: {metrics.L2}");
            Console.WriteLine($"LossFn: {metrics.LossFn}");
            Console.WriteLine($"Rms: {metrics.Rms}");
            Console.WriteLine($"RSquared: {metrics.RSquared}");
            //Rms: 14.6106398994495
            //RSquared: 0.662277823029482
            var score = metrics.Rms - metrics.RSquared;
            double previousHighScore = 0;
            if (File.Exists(modelStatsPath))
            {
                var previousModelData = File.ReadAllLines(modelStatsPath);
                previousHighScore = double.Parse(previousModelData[0]);
            }

            if (score < previousHighScore)
            {
                File.WriteAllText(modelStatsPath, score.ToString() + Environment.NewLine);
                File.AppendAllLines(modelStatsPath, new List<string>
                {
                    $"L1: {metrics.L1:P2}",
                    $"L2: {metrics.L2:P2}",
                    $"LossFn: {metrics.LossFn:P2}",
                    $"Rms: {metrics.Rms:P2}",
                    $"RSquared: {metrics.RSquared:P2}"
                });
                File.AppendAllText(modelStatsPath, "Features:" + Environment.NewLine);
                File.AppendAllLines(modelStatsPath, features);
                File.AppendAllText(modelStatsPath, "Algorithm: " + algorithm.GetType().Name);
                model.WriteAsync(modelPath);
                Console.WriteLine("New model is better");
            }
            else
            {
                Console.WriteLine("Old model is better");
            }

            var prediction = model.Predict(new LegoSetsData()
            {
                ID = 60146,
                Licensed = 0,
                Theme = "City",
                Year = 2017,
                Pieces = 91,
                Minifigs = 1,
                //Price = 9.99f
            });
            Console.WriteLine($"Predicted set price is: {prediction.PredictedPrice}");

            var prediction2 = model.Predict(new LegoSetsData()
            {
                ID = 60148,
                Licensed = 0,
                Theme = "City",
                Year = 2017,
                Pieces = 239,
                Minifigs = 2,
                //Price = 19.99f
            });
            Console.WriteLine($"Predicted set price2 is: {prediction2.PredictedPrice}");

            Console.ReadLine();
        }
    }
}
