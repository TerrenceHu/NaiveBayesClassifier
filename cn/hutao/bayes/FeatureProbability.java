package cn.hutao.bayes;

/**
 * Simple interface defining the method to calculate the feature probability.
 *
 * @param <F> The feature class.
 * @param <C> The category class.
 */
public interface FeatureProbability<F, C> {

    public float featureProbability(F feature, C category);

}
