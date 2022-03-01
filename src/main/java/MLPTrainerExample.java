import dataFiles.MLSandboxDatasets;
import dataFiles.SandboxMLCache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import javax.cache.Cache;
import org.apache.ignite.cache.query.ScanQuery;
import org.apache.ignite.cache.query.QueryCursor;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
//import org.apache.ignite.examples.ExampleNodeStartup;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.LabeledDummyVectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.math.primitives.matrix.Matrix;
import org.apache.ignite.ml.math.primitives.matrix.impl.DenseMatrix;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;
import org.apache.ignite.ml.multiclass.OneVsRestTrainer;
import org.apache.ignite.ml.nn.Activators;
import org.apache.ignite.ml.nn.MLPTrainer;
import org.apache.ignite.ml.clustering.kmeans.KMeansModel;
import org.apache.ignite.ml.clustering.kmeans.KMeansTrainer;
import org.apache.ignite.ml.nn.MultilayerPerceptron;
import org.apache.ignite.ml.nn.UpdatesStrategy;
import org.apache.ignite.ml.nn.architecture.MLPArchitecture;
import org.apache.ignite.ml.optimization.LossFunctions;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDParameterUpdate;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDUpdateCalculator;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.svm.SVMLinearClassificationModel;

import java.io.FileNotFoundException;

/**
 * Example of using distributed {@link MultilayerPerceptron}.
 * <p>
 * Code in this example launches Ignite grid and fills the cache with simple test data.</p>
 * <p>
 * After that it defines a layered architecture and a
 * <a href="https://en.wikipedia.org/wiki/Neural_network">neural network</a> trainer, trains neural network
 * and obtains multilayer perceptron model.</p>
 * <p>
 * Finally, this example loops over the test set, applies the trained model to predict the value and compares prediction
 * to expected outcome.</p>
 * <p>
 * You can change the test data used in this example and re-run it to explore this functionality further.</p>
 * <p>
 * Remote nodes should always be started with special configuration file which enables P2P class loading: {@code
 * 'ignite.{sh|bat} examples/config/example-ignite.xml'}.</p>
 */
public class MLPTrainerExample {
    /**
     * Executes example.
     *
     * @param args Command line arguments, none required.
     */
    public static void main(String[] args) {
        // IMPL NOTE based on MLPGroupTrainerTest#testXOR
        System.out.println(">>> Distributed multilayer perceptron example started.");

        // Start ignite grid.
        try (Ignite ignite = Ignition.start("E:/stuff/MLP_test/config/default-config.xml")) {
            ignite.destroyCache("cacheML");
            System.out.println(">>> Ignite grid started.");

            // Create cache with training data.
            CacheConfiguration<Integer, Vector> trainingSetCfg = new CacheConfiguration<>();
            trainingSetCfg.setName("TRAINING_SET");
            trainingSetCfg.setAffinity(new RendezvousAffinityFunction(false, 10));

//            IgniteCache<Integer, LabeledVector<double[]>> trainingSet = null;
//            IgniteCache<Integer, Vector> dataCache = null;
            IgniteCache<Integer, Vector> dataCache = null;
            try {
//                trainingSet = ignite.createCache(trainingSetCfg);

                // working with dataCache
                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_0_1_2);
//                dataCache = ignite.createCache(trainingSetCfg).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_12);

                Vectorizer<Integer, Vector, Integer, Double> vectorizer =
                        new DummyVectorizer<Integer>().labeled(Vectorizer.LabelCoordinate.FIRST);

                KMeansTrainer trainer = new KMeansTrainer().withAmountOfClusters(3);

                KMeansModel mdl = trainer.fit(
                        ignite,
                        dataCache,
                        vectorizer
                );
                System.out.println(">>> Ended the trainer fitting");

                int totalCnt = dataCache.size();
                int failCnt = 0;
//                double oneFailure = 0;
//                double oneCount = 0;
//                double twoFailure = 0;
//                double twoCount = 0;
//                double zeroFailure = 0;
//                double zeroCount = 0;
//                double threeFailure = 0;
//                double threeCount = 0;

                double[] oneArray = {0.0, 0.0, 0.0, 0.0};
                double[] twoArray = {0.0, 0.0, 0.0, 0.0};
                double[] zeroArray = {0.0, 0.0, 0.0, 0.0};
                double[] threeArray = {0.0, 0.0, 0.0, 0.0};

                // Calculate score.
//                for (int i = 0; i < dataCache.size(); i++) {
//                    LabeledVector<double[]> pnt = dataCache.get(i);
////                    Matrix predicted = mlp.predict(new DenseMatrix(
////                            new double[][] {{
////                                pnt.features().get(0),
////                                    pnt.features().get(1)}}));
////                    Matrix predicted = mlp.predict(new DenseMatrix(
////                                                pnt.features().asArray(), 10));
//                    Matrix predicted = mlp.predict(new DenseMatrix(
////                                                new double[][]{pnt.features().asArray(), {}, {}, {}, {}, {}, {}, {}, {}, {}}));
//                            new double[][]{pnt.features().asArray()}));
////                    Matrix predicted = mlp.predict(pnt.features().toMatrix(true));
//                    double predictedVal = predicted.get(0, 0);
//                    double lbl = pnt.label()[0];
//                    System.out.printf(">>> key: %d\t\t predicted: %.4f\t\tlabel: %.4f\n", i, predictedVal, lbl);
//                    failCnt += Math.abs(predictedVal - lbl) < 0.5 ? 0 : 1;
//                }

                try (QueryCursor<Cache.Entry<Integer, Vector>> observations = dataCache.query(new ScanQuery<>())) {
                    for (Cache.Entry<Integer, Vector> observation : observations) {
                        Vector val = observation.getValue();
                        Vector inputs = val.copyOfRange(1, val.size());
                        double groundTruth = val.get(0);

                        double prediction = mdl.predict(inputs);
                        if (groundTruth == new Double(0)) {
//                            zeroCount++;
//                            zeroFailure += Math.abs(prediction - groundTruth) > 0.5 ? 1 : 0;
                            zeroArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(1)) {
//                            oneCount++;
//                            oneFailure += Math.abs(prediction - groundTruth) > 0.5 ? 1 : 0;
                            oneArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(2)) {
//                            twoCount++;
//                            twoFailure += Math.abs(prediction - groundTruth) > 0.5 ? 1 : 0;
                            twoArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(3)) {
//                            threeCount++;
//                            threeFailure += Math.abs(prediction - groundTruth) > 0.5 ? 1 : 0;
                            threeArray[(int) prediction] += 1;
                        }
                        System.out.printf(">>> | %.4f\t\t\t| %.4f\t\t|\n", prediction, groundTruth);
                    }

//                    System.out.print("The zero cluster results\n");
//                    System.out.print("The number of errors is\t");
//                    System.out.println(zeroFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(zeroFailure/zeroCount);
//                    System.out.println();
//
//                    System.out.print("The one cluster results\n");
//                    System.out.print("The number of errors is\t");
//                    System.out.println(oneFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(oneFailure/oneCount);
//                    System.out.println();
//
//                    System.out.print("The two cluster results\n");
//                    System.out.print("The number of errors is\t");
//                    System.out.println(twoFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(twoFailure/twoCount);
//                    System.out.println();

                    System.out.println("The zero distribution");
                    for(int i=0; i<zeroArray.length; i++){
                        System.out.printf("%.1f",zeroArray[i]);
                        System.out.print("   ");
                    }
//                    System.out.print("The number of errors is\t");
//                    System.out.println(zeroFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(zeroFailure/zeroCount);
                    System.out.println();

                    System.out.println("The one distribution");
                    for(int i=0; i<oneArray.length; i++){
                        System.out.printf("%.1f",oneArray[i]);
                        System.out.print("   ");
                    }
//                    System.out.print("The number of errors is\t");
//                    System.out.println(oneFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(oneFailure/oneCount);
                    System.out.println();

                    System.out.println("The two distribution");
                    for(int i=0; i<twoArray.length; i++){
                        System.out.printf("%.1f",twoArray[i]);
                        System.out.print("   ");
                    }
//                    System.out.print("The number of errors is\t");
//                    System.out.println(twoFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(twoFailure/twoCount);
                    System.out.println();

                    System.out.println("The three distribution");
                    for(int i=0; i<twoArray.length; i++){
                        System.out.printf("%.1f",threeArray[i]);
                        System.out.print("   ");
                    }
//                    System.out.print("The number of errors is\t");
//                    System.out.println(threeFailure);
//                    System.out.print("The error percentage is\t");
//                    System.out.println(threeFailure/threeCount);
                    System.out.println();


                    System.out.println(">>> ---------------------------------");
                    System.out.println(">>> KMeans clustering algorithm over cached dataset usage example completed.");
                }

//                double failRatio = (double) failCnt / totalCnt;
//
//                System.out.println("\n>>> Fail percentage: " + (failRatio * 100) + "%.");
//                System.out.println("\n>>> Distributed multilayer perceptron example completed.");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } finally {
                System.out.println("\n>>> TaDAA.");
            }
        } finally {
            System.out.flush();
            System.exit(0);
        }
    }
}