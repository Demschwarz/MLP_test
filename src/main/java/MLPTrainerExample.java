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

            IgniteCache<Integer, Vector> dataCache = null;
            try {
                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_0_1_2);

                Vectorizer<Integer, Vector, Integer, Double> vectorizer =
                        new DummyVectorizer<Integer>().labeled(Vectorizer.LabelCoordinate.FIRST);

                //Явно задаём количество кластеров

                KMeansTrainer trainer = new KMeansTrainer().withAmountOfClusters(2);

                KMeansModel mdl = trainer.fit(
                        ignite,
                        dataCache,
                        vectorizer
                );
                System.out.println(">>> Ended the trainer fitting");


                double[] oneArray = {0.0, 0.0};
                double[] twoArray = {0.0, 0.0};
                double[] zeroArray = {0.0, 0.0};
                double[] threeArray = {0.0, 0.0};
                double[] fourArray = {0.0, 0.0};
                double[] fiveArray = {0.0, 0.0};
                double[] sixArray = {0.0, 0.0};
                double[] sevenArray = {0.0, 0.0};
                double[] eightArray = {0.0, 0.0};
                double[] nineArray = {0.0, 0.0};


                try (QueryCursor<Cache.Entry<Integer, Vector>> observations = dataCache.query(new ScanQuery<>())) {
                    for (Cache.Entry<Integer, Vector> observation : observations) {
                        Vector val = observation.getValue();
                        Vector inputs = val.copyOfRange(1, val.size());
                        double groundTruth = val.get(0);

                        double prediction = mdl.predict(inputs);
                        if (groundTruth == new Double(0)) {
                            zeroArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(1)) {
                            oneArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(2)) {
                            twoArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(3)) {
                            threeArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(4)) {
                            fourArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(5)) {
                            fiveArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(6)) {
                            sixArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(7)) {
                            sevenArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(8)) {
                            eightArray[(int) prediction] += 1;
                        }
                        if (groundTruth == new Double(9)) {
                            nineArray[(int) prediction] += 1;
                        }
                        System.out.printf(">>> | %.4f\t\t\t| %.4f\t\t|\n", prediction, groundTruth);
                    }

                    System.out.println("The zero distribution");
                    for(int i=0; i<zeroArray.length; i++){
                        System.out.printf("%.1f",zeroArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The one distribution");
                    for(int i=0; i<oneArray.length; i++){
                        System.out.printf("%.1f",oneArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The two distribution");
                    for(int i=0; i<twoArray.length; i++){
                        System.out.printf("%.1f",twoArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The three distribution");
                    for(int i=0; i<threeArray.length; i++){
                        System.out.printf("%.1f",threeArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The four distribution");
                    for(int i=0; i<fourArray.length; i++){
                        System.out.printf("%.1f",fourArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The five distribution");
                    for(int i=0; i<fiveArray.length; i++){
                        System.out.printf("%.1f",fiveArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The six distribution");
                    for(int i=0; i<sixArray.length; i++){
                        System.out.printf("%.1f",sixArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The seven distribution");
                    for(int i=0; i<sevenArray.length; i++){
                        System.out.printf("%.1f",sevenArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The eight distribution");
                    for(int i=0; i<eightArray.length; i++){
                        System.out.printf("%.1f",eightArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();

                    System.out.println("The nine distribution");
                    for(int i=0; i<nineArray.length; i++){
                        System.out.printf("%.1f",nineArray[i]);
                        System.out.print("   ");
                    }
                    System.out.println();


                    System.out.println(">>> ---------------------------------");
                    System.out.println(">>> KMeans clustering algorithm over cached dataset usage example completed.");
                }
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