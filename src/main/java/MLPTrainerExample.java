import dataFiles.MLSandboxDatasets;
import dataFiles.SandboxMLCache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
//import org.apache.ignite.examples.ExampleNodeStartup;
import org.apache.ignite.ml.dataset.feature.extractor.impl.LabeledDummyVectorizer;
import org.apache.ignite.ml.math.primitives.matrix.Matrix;
import org.apache.ignite.ml.math.primitives.matrix.impl.DenseMatrix;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;
import org.apache.ignite.ml.nn.Activators;
import org.apache.ignite.ml.nn.MLPTrainer;
import org.apache.ignite.ml.nn.MultilayerPerceptron;
import org.apache.ignite.ml.nn.UpdatesStrategy;
import org.apache.ignite.ml.nn.architecture.MLPArchitecture;
import org.apache.ignite.ml.optimization.LossFunctions;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDParameterUpdate;
import org.apache.ignite.ml.optimization.updatecalculators.SimpleGDUpdateCalculator;
import org.apache.ignite.ml.structures.LabeledVector;

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
            CacheConfiguration<Integer, LabeledVector<double[]>> trainingSetCfg = new CacheConfiguration<>();
            trainingSetCfg.setName("TRAINING_SET");
            trainingSetCfg.setAffinity(new RendezvousAffinityFunction(false, 10));

//            IgniteCache<Integer, LabeledVector<double[]>> trainingSet = null;
//            IgniteCache<Integer, Vector> dataCache = null;
            IgniteCache<Integer, LabeledVector<double[]>> dataCache = null;
            try {
//                trainingSet = ignite.createCache(trainingSetCfg);

                // working with dataCache
                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_200);
//                dataCache = ignite.createCache(trainingSetCfg).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_12);

//                // Fill cache with training data.
//                trainingSet.put(0, new LabeledVector<>(VectorUtils.of(0, 0), new double[] {0}));
//                trainingSet.put(1, new LabeledVector<>(VectorUtils.of(0, 1), new double[] {1}));
//                trainingSet.put(2, new LabeledVector<>(VectorUtils.of(1, 0), new double[] {1}));
//                trainingSet.put(3, new LabeledVector<>(VectorUtils.of(1, 1), new double[] {0}));

                // Define a layered architecture.
                MLPArchitecture arch = new MLPArchitecture(784).
                        withAddedLayer(9, true, Activators.RELU).
                        withAddedLayer(1, false, Activators.RELU);

                // Define a neural network trainer.
                MLPTrainer<SimpleGDParameterUpdate> trainer = new MLPTrainer<>(
                        arch,
                        LossFunctions.MSE,
                        new UpdatesStrategy<>(
                                new SimpleGDUpdateCalculator(0.1),
                                SimpleGDParameterUpdate.SUM_LOCAL,
                                SimpleGDParameterUpdate.AVG
                        ),
                        30000,
                        4,
                        50,
                        123L
                );

                // Train neural network and get multilayer perceptron model.
//                MultilayerPerceptron mlp = trainer.fit(ignite, trainingSet, new LabeledDummyVectorizer<>());
                System.out.println(">>> Started the trainer fitting");
                MultilayerPerceptron mlp = trainer.fit(ignite, dataCache, new LabeledDummyVectorizer<>());
                System.out.println(">>> Ended the trainer fitting");

                int totalCnt = dataCache.size();
                int failCnt = 0;

                // Calculate score.
                for (int i = 0; i < dataCache.size(); i++) {
                    LabeledVector<double[]> pnt = dataCache.get(i);
//                    Matrix predicted = mlp.predict(new DenseMatrix(
//                            new double[][] {{
//                                pnt.features().get(0),
//                                    pnt.features().get(1)}}));
//                    Matrix predicted = mlp.predict(new DenseMatrix(
//                                                pnt.features().asArray(), 1));
                    Matrix predicted = mlp.predict(pnt.features().toMatrix(true));
                    double predictedVal = predicted.get(0, 0);
                    double lbl = pnt.label()[0];
                    System.out.printf(">>> key: %d\t\t predicted: %.4f\t\tlabel: %.4f\n", i, predictedVal, lbl);
                    failCnt += Math.abs(predictedVal - lbl) < 0.5 ? 0 : 1;
                }

                double failRatio = (double)failCnt / totalCnt;

                System.out.println("\n>>> Fail percentage: " + (failRatio * 100) + "%.");
                System.out.println("\n>>> Distributed multilayer perceptron example completed.");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } finally {
                System.out.println("\n>>> TaDAA.");
            }
        }
        finally {
            System.out.flush();
        }
    }
}