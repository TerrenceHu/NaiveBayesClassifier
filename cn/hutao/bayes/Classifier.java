package cn.hutao.bayes;

import java.util.Collection;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/**
 * Abstract base extended by any concrete classifier.  It implements the basic
 * functionality for storing categories or features and can be used to calculate
 * basic probabilities – both category and feature probabilities. The classify
 * function has to be implemented by the concrete classifier class.
 *
 * @param <F> A feature class
 * @param <C> A category class
 */
public abstract class Classifier<F, C> implements FeatureProbability<F, C> {

    /**
     * Initial capacity of category dictionaries.
     */
    private static final int INITIAL_CATEGORY_DICTIONARY_CAPACITY = 8;

    /**
     * Initial capacity of feature dictionaries.
     */
    private static final int INITIAL_FEATURE_DICTIONARY_CAPACITY = 1024;

    /**
     * The initial memory capacity or how many classifications are memorized.
     */
    private int memoryCapacity = 1000;

    /**
     * A dictionary mapping features to their number of occurrences in each
     * known category.
     */
    private Dictionary<C, Dictionary<F, Integer>> featureCountPerCategory;

    /**
     * A dictionary mapping features to their number of occurrences.
     */
    private Dictionary<F, Integer> totalFeatureCount;

    /**
     * A dictionary mapping categories to their number of occurrences.
     */
    private Dictionary<C, Integer> totalCategoryCount;

    /**
     * The classifier's memory. It will forget old classifications as soon as
     * they become too old.
     */
    private Queue<Classification<F, C>> memoryQueue;

    /**
     * Constructs a new classifier without any trained knowledge.
     */
    public Classifier() {
        this.reset();
    }

    /**
     * Resets the learned feature and category counts.
     */
    public void reset() {
        this.featureCountPerCategory =
                new Hashtable<C, Dictionary<F,Integer>>(
                        Classifier.INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        this.totalFeatureCount =
                new Hashtable<F, Integer>(
                        Classifier.INITIAL_FEATURE_DICTIONARY_CAPACITY);
        this.totalCategoryCount =
                new Hashtable<C, Integer>(
                        Classifier.INITIAL_CATEGORY_DICTIONARY_CAPACITY);
        this.memoryQueue = new LinkedList<Classification<F, C>>();
    }

    /**
     * Returns a Set of features the classifier knows about.
     *
     * @return The Set of features the classifier knows about.
     */
    public Set<F> getFeatures() {
        return ((Hashtable<F, Integer>) this.totalFeatureCount).keySet();
    }

    /**
     * Returns a Set of categories the classifier knows about.
     *
     * @return The Set of categories the classifier knows about.
     */
    public Set<C> getCategories() {
        return ((Hashtable<C, Integer>) this.totalCategoryCount).keySet();
    }

    /**
     * Retrieves the memory's capacity.
     *
     * @return The memory's capacity.
     */
    public int getMemoryCapacity() {
        return memoryCapacity;
    }

    /**
     * Sets the memory's capacity.  If the new value is less than the old
     * value, the memory will be truncated accordingly.
     *
     * @param memoryCapacity The new memory capacity.
     */
    public void setMemoryCapacity(int memoryCapacity) {
        for (int i = this.memoryCapacity; i > memoryCapacity; i--) {
            this.memoryQueue.poll();
        }
        this.memoryCapacity = memoryCapacity;
    }

    /**
     * Increments the count of a given feature in the given category.  This is
     * equal to telling the classifier, that this feature has occurred in this
     * category.
     *
     * @param feature The feature, which count to increase.
     * @param category The category the feature occurred in.
     */
    public void incrementFeature(F feature, C category) {
        Dictionary<F, Integer> features = this.featureCountPerCategory.get(category);
        if (features == null) {
            this.featureCountPerCategory.put(category,
                    new Hashtable<F, Integer>(
                            Classifier.INITIAL_FEATURE_DICTIONARY_CAPACITY));
            features = this.featureCountPerCategory.get(category);
        }
        Integer count = features.get(feature);
        if (count == null) {
            features.put(feature, 0);
            count = features.get(feature);
        }
        features.put(feature, ++count);

        Integer totalCount = this.totalFeatureCount.get(feature);
        if (totalCount == null) {
            this.totalFeatureCount.put(feature, 0);
            totalCount = this.totalFeatureCount.get(feature);
        }
        this.totalFeatureCount.put(feature, ++totalCount);
    }

    /**
     * Increments the count of a given category.  This is equal to telling the
     * classifier, that this category has occurred once more.
     *
     * @param category The category, which count to increase.
     */
    public void incrementCategory(C category) {
        Integer count = this.totalCategoryCount.get(category);
        if (count == null) {
            this.totalCategoryCount.put(category, 0);
            count = this.totalCategoryCount.get(category);
        }
       this.totalCategoryCount.put(category, ++count);
    }

    /**
     * Decrements the count of a given feature in the given category.  This is
     * equal to telling the classifier that this feature was classified once in
     * the category.
     *
     * @param feature The feature to decrement the count for.
     * @param category The category.
     */
    public void decrementFeature(F feature, C category) {
        Dictionary<F, Integer> features =
                this.featureCountPerCategory.get(category);
        if (features == null) {
            return;
        }
        Integer count = features.get(feature);
        if (count == null) {
            return;
        }
        if (count.intValue() == 1) {
            features.remove(feature);
            if (features.size() == 0) {
                this.featureCountPerCategory.remove(category);
            }
        } else {
            features.put(feature, --count);
        }

        Integer totalCount = this.totalFeatureCount.get(feature);
        if (totalCount == null) {
            return;
        }
        if (totalCount.intValue() == 1) {
            this.totalFeatureCount.remove(feature);
        } else {
            this.totalFeatureCount.put(feature, --totalCount);
        }
    }

    /**
     * Decrements the count of a given category.  This is equal to telling the
     * classifier, that this category has occurred once less.
     *
     * @param category The category, which count to increase.
     */
    public void decrementCategory(C category) {
        Integer count = this.totalCategoryCount.get(category);
        if (count == null) {
            return;
        }
        if (count.intValue() == 1) {
            this.totalCategoryCount.remove(category);
        } else {
            this.totalCategoryCount.put(category, --count);
        }
    }

    /**
     * Retrieves the total number of categories the classifier knows about.
     *
     * @return The total category count.
     */
    public int getCategoriesTotal() {
        int toReturn = 0;
        for (Enumeration<Integer> e = this.totalCategoryCount.elements();
             e.hasMoreElements();) {
            toReturn += e.nextElement();
        }
        return toReturn;
    }

    /**
     * Retrieves the number of occurrences of the given feature in the given
     * category.
     *
     * @param feature The feature, which count to retrieve.
     * @param category The category, which the feature occurred in.
     * @return The number of occurrences of the feature in the category.
     */
    public int featureCount(F feature, C category) {
        Dictionary<F, Integer> features =
                this.featureCountPerCategory.get(category);
        if (features == null)
            return 0;
        Integer count = features.get(feature);
        return (count == null) ? 0 : count.intValue();
    }

    /**
     * Retrieves the number of occurrences of the given category.
     * 
     * @param category The category, which count should be retrieved.
     * @return The number of occurrences.
     */
    public int categoryCount(C category) {
        Integer count = this.totalCategoryCount.get(category);
        return (count == null) ? 0 : count.intValue();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float featureProbability(F feature, C category) {
        if (this.categoryCount(category) == 0) {
            return 0;
        }
        return (float) this.featureCount(feature, category)
                / (float) this.categoryCount(category);
    }

    /**
     * Retrieves the weighed average P(feature|category) with
     * overall weight of 1.0 and an assumed probability of 0.5.
     * The probability defaults to the overall feature probability.
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(F feature, C category) {
        return this.featureWeighedAverage(feature, category, null, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average P(feature|category) with
     * overall weight of 1.0, an assumed probability of
     * 0.5 and the given object to use for probability calculation.
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(F feature, C category,
                                       FeatureProbability<F, C> calculator) {
        return this.featureWeighedAverage(feature, category, calculator, 1.0f, 0.5f);
    }

    /**
     * Retrieves the weighed average P(feature|category) with
     * the given weight and an assumed probability of 0.5 and the
     * given object to use for probability calculation.
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @param weight The feature weight.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(F feature, C category,
                                       FeatureProbability<F, C> calculator, float weight) {
        return this.featureWeighedAverage(feature, category,
                calculator, weight, 0.5f);
    }

    /**
     * Retrieves the weighed average P(feature|category) with
     * the given weight, the given assumed probability and the given object to
     * use for probability calculation.
     *
     * @param feature The feature, which probability to calculate.
     * @param category The category.
     * @param calculator The calculating object.
     * @param weight The feature weight.
     * @param assumedProbability The assumed probability.
     * @return The weighed average probability.
     */
    public float featureWeighedAverage(F feature, C category,
                                       FeatureProbability<F, C> calculator, float weight,
                                       float assumedProbability) {

        /*
         * use the given calculating object or the default method to calculate
         * the probability that the given feature occurred in the given
         * category.
         */
        final float basicProbability =
                (calculator == null)
                    ? this.featureProbability(feature, category)
                            : calculator.featureProbability(feature, category);

        Integer totals = this.totalFeatureCount.get(feature);
        if (totals == null)
            totals = 0;
        return (weight * assumedProbability + totals  * basicProbability)
                / (weight + totals);
    }

    /**
     * Train the classifier.
     *
     * @param category The category the features belong to.
     * @param features The features that resulted in the given category.
     */
    public void learn(C category, Collection<F> features) {
        this.learn(new Classification<F, C>(features, category));
    }

    /**
     * Train the classifier.
     *
     * @param classification The classification to learn.
     */
    public void learn(Classification<F, C> classification) {

        for (F feature : classification.getFeatureset()) {
            this.incrementFeature(feature, classification.getCategory());
        }
        this.incrementCategory(classification.getCategory());

        this.memoryQueue.offer(classification);
        if (this.memoryQueue.size() > this.memoryCapacity) {
            Classification<F, C> toForget = this.memoryQueue.remove();

            for (F feature : toForget.getFeatureset()) {
                this.decrementFeature(feature, toForget.getCategory());
            }
            this.decrementCategory(toForget.getCategory());
        }
    }

    /**
     * The classify method.
     *
     * @param features The features to classify.
     * @return The category most likely.
     */
    public abstract Classification<F, C> classify(Collection<F> features);

}