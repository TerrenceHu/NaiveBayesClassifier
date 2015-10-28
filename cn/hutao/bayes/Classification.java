package cn.hutao.bayes;

import java.util.Collection;

/**
 * A basic wrapper reflecting a classification.
 *
 * @param <F> The feature class.
 * @param <C> The category class.
 */
public class Classification<F, C> {

    /**
     * The classified featureset.
     */
    private Collection<F> featureset;

    /**
     * The category as which the featureset was classified.
     */
    private C category;

    /**
     * The probability that the featureset belongs to the given category.
     */
    private float probability;

    /**
     * Constructs a new Classification with the parameters given.
     *
     * @param featureset The featureset.
     * @param category The category.
     * @param probability The probability.
     */
    public Classification(Collection<F> featureset, C category, float probability) {
        this.featureset = featureset;
        this.category = category;
        this.probability = probability;
    }

    /**
     * Constructs a new Classification with the parameters given
     * and a default probability of 1.
     *
     * @param featureset The featureset.
     * @param category The category.
     */
    public Classification(Collection<F> featureset, C category) {
        this(featureset, category, 1.0f);
    }

    /**
     * Retrieves the featureset.
     *
     * @return The featureset.
     */
    public Collection<F> getFeatureset() {
        return featureset;
    }

    /**
     * Retrieves the category.
     *
     * @return The category.
     */
    public C getCategory() {
        return category;
    }


    /**
     * Retrieves the classification's probability.
     * @return
     */
    public float getProbability() {
        return this.probability;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "Classification [category=" + this.category
                + ", probability=" + this.probability
                + ", featureset=" + this.featureset + "]";
    }

}
