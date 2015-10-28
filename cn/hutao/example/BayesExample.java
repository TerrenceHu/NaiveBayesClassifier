package cn.hutao.example;

import java.util.Arrays;

import cn.hutao.bayes.BayesClassifier;
import cn.hutao.bayes.Classifier;

public class BayesExample {

    public static void main(String[] args) {

        Classifier<String, String> bayes = new BayesClassifier<String, String>();

        String[] positiveText = "I love sunny days".split("\\s");
        bayes.learn("pos", Arrays.asList(positiveText));

        String[] negativeText = "I hate rain".split("\\s");
        bayes.learn("neg", Arrays.asList(negativeText));

        String[] unknownText1 = "today is a sunny day".split("\\s");
        String[] unknownText2 = "there will be rain".split("\\s");

        System.out.println(bayes.classify(Arrays.asList(unknownText1)).getCategory());
        System.out.println(bayes.classify(Arrays.asList(unknownText2)).getCategory());

        /*
         * The BayesClassifier extends the abstract Classifier and provides
         * detailed classification results that can be retrieved by calling
         * the classifyDetailed Method.
         *
         * The classification with the highest probability is the resulting
         * classification. The returned List will look like this.
         * [
         *   Classification [
         *     category=negative,
         *     probability=0.0078125,
         *     featureset=[today, is, a, sunny, day]
         *   ],
         *   Classification [
         *     category=positive,
         *     probability=0.0234375,
         *     featureset=[today, is, a, sunny, day]
         *   ]
         * ]
         */
        ((BayesClassifier<String, String>) bayes).classifyDetailed(Arrays.asList(unknownText1));

        /*
         * Please note, that this particular classifier implementation will
         * "forget" learned classifications after a few learning sessions. The
         * number of learning sessions it will record can be set as follows:
         */
        bayes.setMemoryCapacity(500); // remember the last 500 learned classifications
    }

}
