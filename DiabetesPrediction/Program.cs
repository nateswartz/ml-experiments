using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace DiabetesPrediction
{
    class Program
    {
        static PredictionModel<DiabetesData, DiabetesPrediction> model;
        static string modelPath = "mymodel.zip";

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

        static void Main(string[] args)
        {
            Console.WriteLine("Creating new model");
            var pipeline = new LearningPipeline();

            string dataPath = "diabetes-data-training.txt";
            pipeline.Add(new TextLoader<DiabetesData>(dataPath, separator: ","));

            var features = new string[] { "BMI", "Age", "Pregnancies", "PlasmaGlucoseConcentration", "TricepsSkinFoldThickness" };
            pipeline.Add(new ColumnConcatenator("Features", features));

            var algorithm = new AveragedPerceptronBinaryClassifier();
            pipeline.Add(algorithm);

            model = pipeline.Train<DiabetesData, DiabetesPrediction>();

            var testDataPath = "diabetes-data-test.txt";
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
            if (File.Exists("modelstats.txt"))
            {
                var previousModelData = File.ReadAllLines("modelstats.txt");
                previousHighScore = double.Parse(previousModelData[0]);
            }

            if (score > previousHighScore)
            {
                File.WriteAllText("modelstats.txt", score.ToString() + Environment.NewLine);
                File.AppendAllLines("modelstats.txt", new List<string>
                {
                    $"Accuracy: {metrics.Accuracy:P2}",
                    $"Auc: {metrics.Auc:P2}",
                    $"F1Score: {metrics.F1Score:P2}"
                });
                File.AppendAllText("modelstats.txt", "Features:" + Environment.NewLine);
                File.AppendAllLines("modelstats.txt", features);
                File.AppendAllText("modelstats.txt", "Algorithm: " + algorithm.GetType().Name);
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