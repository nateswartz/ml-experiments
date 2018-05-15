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
    public class DiabetesData
    {
        [Column("0")]
        public float Pregnancies;

        [Column("1")]
        public float PlasmaGlucoseConcentration;

        [Column("2")]
        public float DiastolicBloodPressure;

        [Column("3")]
        public float TricepsSkinFoldThickness;

        [Column("4")]
        public float TwoHourSerumInsulin;

        [Column("5")]
        public float BMI;

        [Column("6")]
        public float DiabetesPedigreeFunction;

        [Column("7")]
        public float Age;

        [Column("8")]
        [ColumnName("Label")]
        public float Label;
    }

    public class DiabetesPrediction
    {
        [ColumnName("Label")]
        public float PredictedLabel;
    }

    public class DiabetesExperiment
    {
        static PredictionModel<DiabetesData, DiabetesPrediction> model;
        static string modelPath = "Data/diabetes_model.zip";
        static string modelStatsPath = "Data/diabetes_modelstats.txt";
        static string dataPath = "Data/diabetes-data-training.txt";
        static string testDataPath = "Data/diabetes-data-test.txt";

        public static void Execute()
        {
            Console.WriteLine("Executing Diabetes Experiment");
            Console.WriteLine("Creating new model");
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader<DiabetesData>(dataPath, separator: ","));

            var features = new string[] { "BMI", "Age", "Pregnancies", "PlasmaGlucoseConcentration", "TricepsSkinFoldThickness" };
            pipeline.Add(new ColumnConcatenator("Features", features));

            var algorithm = new BinaryLogisticRegressor();
            pipeline.Add(algorithm);

            model = pipeline.Train<DiabetesData, DiabetesPrediction>();

            var testData = new TextLoader<DiabetesData>(testDataPath, separator: ",");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

            var score = metrics.Accuracy + metrics.Auc + metrics.F1Score;
            double previousHighScore = 0;
            if (File.Exists(modelStatsPath))
            {
                var previousModelData = File.ReadAllLines(modelStatsPath);
                previousHighScore = double.Parse(previousModelData[0]);
            }

            if (score > previousHighScore)
            {
                File.WriteAllText(modelStatsPath, score.ToString() + Environment.NewLine);
                File.AppendAllLines(modelStatsPath, new List<string>
                {
                    $"Accuracy: {metrics.Accuracy:P2}",
                    $"Auc: {metrics.Auc:P2}",
                    $"F1Score: {metrics.F1Score:P2}"
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
            Console.ReadLine();
        }
    }
}
