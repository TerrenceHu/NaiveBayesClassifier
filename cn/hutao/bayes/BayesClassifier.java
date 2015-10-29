package cn.hutao.bayes;

import java.util.Collection;
import java.util.Comparator;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * A concrete implementation of the abstract Classifier class.  The Bayes
 * classifier implements a naive Bayes approach to classifying a given set of
 * features: classify(feat1,...,featN) = argmax(P(cat)*PROD(P(featI|cat)
 *
 * @param <F> The feature class.
 * @param <C> The category class.
 */
public class BayesClassifier<F, C> extends Classifier<F, C> {

    /**
     * Calculates the product of all feature probabilities: PROD(P(featI|cat)
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The product of all feature probabilities.
     */
    private double featuresProbabilityLogSum(Collection<F> features, C category) {
        double logSum = 1.0f;
        for (F feature : features) {
            logSum += Math.log(this.featureWeighedAverage(feature, category));
        }
        return logSum;
    }

    /**
     * Calculates the probability that the features can be classified as the
     * category given.
     *
     * @param features The set of features to use.
     * @param category The category to test for.
     * @return The probability that the features can be classified as the
     *    category.
     */
    private double categoryProbability(Collection<F> features, C category) {
        return Math.log(((double) this.categoryCount(category)
                    / (double) this.getCategoriesTotal()))
                + featuresProbabilityLogSum(features, category);
    }

    /**
     * Retrieves a sorted Set of probabilities that the given set
     * of features is classified as the available categories.
     *
     * @param features The set of features to use.
     * @return A sorted Set of category-probability-entries.
     */
    private SortedSet<Classification<F, C>> categoryProbabilities(Collection<F> features) {

        /*
         * Sort the set according to the possibilities. Because we have to sort
         * by the mapped value and not by the mapped key, we can not use a
         * sorted tree (TreeMap) and we have to use a set-entry approach to
         * achieve the desired functionality. A custom comparator is therefore
         * needed.
         */
        SortedSet<Classification<F, C>> probabilities =
                new TreeSet<Classification<F, C>>(new Comparator<Classification<F, C>>() {

                    @Override
                    public int compare(Classification<F, C> o1, Classification<F, C> o2) {
                        int toReturn = Double.compare(o1.getProbability(), o2.getProbability());
                        if ((toReturn == 0) && !o1.getCategory().equals(o2.getCategory())) {
                            toReturn = -1;
                        }
                        return toReturn;
                    }
                });

        for (C category : this.getCategories()) {
            probabilities.add(new Classification<F, C>(
                    features, category, this.categoryProbability(features, category)));
        }
        return probabilities;
    }

    /**
     * Classifies the given set of features.
     *
     * @return The category the set of features is classified as.
     */
    @Override
    public Classification<F, C> classify(Collection<F> features) {
        SortedSet<Classification<F, C>> probabilites = this.categoryProbabilities(features);

        if (probabilites.size() > 0) {
            return probabilites.last();
        }
        return null;
    }

    /**
     * Classifies the given set of features. and return the full details of the
     * classification.
     *
     * @return The set of categories the set of features is classified as.
     */
    public Collection<Classification<F, C>> classifyDetailed(Collection<F> features) {
        return this.categoryProbabilities(features);
    }

}
