package cn.hutao.example;

import java.util.Arrays;
import java.util.SortedSet;

import cn.hutao.bayes.BayesClassifier;
import cn.hutao.bayes.Classification;
import cn.hutao.bayes.Classifier;

public class BayesExample {

    public static void main(String[] args) {

        Classifier<String, String> bayes = new BayesClassifier<String, String>();

        bayes.setMemoryCapacity(500);

        String[] positiveText = "I love sunny days".split("\\s");
        bayes.learn("pos", Arrays.asList(positiveText));

        String[] negativeText = "I hate rain".split("\\s");
        bayes.learn("neg", Arrays.asList(negativeText));

        String[] unknownText1 = "today is a sunny day".split("\\s");
        String[] unknownText2 = "there will be rain".split("\\s");

        System.out.println(bayes.classify(Arrays.asList(unknownText1)).getCategory());

        SortedSet<Classification<String, String>> probabilites =
                (SortedSet<Classification<String, String>>)
                        ((BayesClassifier<String, String>) bayes).classifyDetailed(Arrays.asList(unknownText1));
        System.out.print(probabilites + "\n");

        System.out.println(bayes.classify(Arrays.asList(unknownText2)).getCategory());

        probabilites =
                (SortedSet<Classification<String, String>>)
                        ((BayesClassifier<String, String>) bayes).classifyDetailed(Arrays.asList(unknownText2));
        System.out.print(probabilites + "\n");
    }

}
