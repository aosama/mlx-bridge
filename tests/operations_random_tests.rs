//! Tier 4 marshalling evidence for the random family: keys and distribution
//! arguments reach the generator. The generators' statistical quality belongs
//! to MLX's own suite, so nothing here samples at scale.

#[path = "support/bridge_fixtures.rs"]
mod bridge_fixtures;

use bridge_fixtures::{float_array, floats_of, gpu_context, ints_of};
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxDtype};

fn uniform_sample(context: &MlxBindingsContext, key: Option<&MlxArray>) -> Vec<f32> {
    let low = MlxArray::from_scalar_f32(0.0);
    let high = MlxArray::from_scalar_f32(1.0);
    let sample = context
        .random_uniform(&low, &high, &[8], MlxDtype::Float32, key)
        .expect("the sample should build");
    floats_of(&sample)
}

#[test]
fn should_reproduce_a_sample_from_the_same_key() {
    let context = gpu_context();

    let first_key = context.random_key(42).unwrap();
    let second_key = context.random_key(42).unwrap();

    assert_eq!(
        uniform_sample(&context, Some(&first_key)),
        uniform_sample(&context, Some(&second_key))
    );
}

#[test]
fn should_produce_different_samples_from_different_keys() {
    let context = gpu_context();

    let first_key = context.random_key(1).unwrap();
    let second_key = context.random_key(2).unwrap();

    assert_ne!(
        uniform_sample(&context, Some(&first_key)),
        uniform_sample(&context, Some(&second_key))
    );
}

#[test]
fn should_reproduce_keyless_samples_after_reseeding_the_global_generator() {
    let context = gpu_context();

    context.seed_random_generation(7).unwrap();
    let first_sample = uniform_sample(&context, None);
    let following_sample = uniform_sample(&context, None);
    context.seed_random_generation(7).unwrap();
    let reseeded_sample = uniform_sample(&context, None);

    assert_eq!(first_sample, reseeded_sample);
    assert_ne!(first_sample, following_sample);
}

#[test]
fn should_split_a_key_into_independent_keys() {
    let context = gpu_context();
    let parent_key = context.random_key(5).unwrap();

    let (first_key, second_key) = context.split_random_key(&parent_key).unwrap();
    let many_keys = context.random_split_num(&parent_key, 4).unwrap();

    assert_ne!(
        first_key.to_vec_u32().unwrap(),
        second_key.to_vec_u32().unwrap()
    );
    assert_eq!(many_keys.shape(), vec![4, 2]);
}

#[test]
fn should_keep_integer_samples_inside_the_half_open_range() {
    let context = gpu_context();
    let key = context.random_key(17).unwrap();
    let low = MlxArray::from_scalar_i32(3);
    let high = MlxArray::from_scalar_i32(6);

    let sample = context
        .random_randint(&low, &high, &[256], MlxDtype::Int32, Some(&key))
        .unwrap();

    for value in ints_of(&sample) {
        assert!((3..6).contains(&value), "{value} escaped the range");
    }
}

#[test]
fn should_sample_a_near_certain_category() {
    let context = gpu_context();
    let key = context.random_key(29).unwrap();
    let logits = float_array(&[-100.0, 100.0, -100.0], &[3]);

    let sample = context
        .categorical_sample_with_key(&logits, 0, &key)
        .unwrap();

    assert_eq!(sample.to_vec_u32().unwrap(), vec![1]);
}

#[test]
fn should_reproduce_a_categorical_sample_from_the_same_seed() {
    let context = gpu_context();
    let logits = float_array(&[0.0, 0.0, 0.0, 0.0], &[4]);

    let first_sample = context.categorical_sample(&logits, 0, 31).unwrap();
    let second_sample = context.categorical_sample(&logits, 0, 31).unwrap();

    assert_eq!(
        first_sample.to_vec_u32().unwrap(),
        second_sample.to_vec_u32().unwrap()
    );
}

#[test]
fn should_forward_uniform_bounds_in_order() {
    let context = gpu_context();
    let key = context.random_key(11).unwrap();
    let low = MlxArray::from_scalar_f32(2.0);
    let high = MlxArray::from_scalar_f32(4.0);

    let sample = context
        .random_uniform(&low, &high, &[64], MlxDtype::Float32, Some(&key))
        .unwrap();

    for value in floats_of(&sample) {
        assert!((2.0..4.0).contains(&value), "{value} escaped the bounds");
    }
}

#[test]
fn should_forward_the_normal_mean_and_deviation_in_order() {
    let context = gpu_context();
    let key = context.random_key(13).unwrap();

    // A zero deviation makes the answer exact, so swapped arguments would show
    // without any statistical test of MLX's generator.
    let sample = context
        .random_normal(&[2], MlxDtype::Float32, 5.0, 0.0, &key)
        .unwrap();

    assert_eq!(floats_of(&sample), vec![5.0, 5.0]);
}

#[test]
fn should_forward_the_bernoulli_probability() {
    let context = gpu_context();
    let key = context.random_key(19).unwrap();

    let never = context
        .random_bernoulli(&MlxArray::from_scalar_f32(0.0), &[4], Some(&key))
        .unwrap();
    let always = context
        .random_bernoulli(&MlxArray::from_scalar_f32(1.0), &[4], Some(&key))
        .unwrap();

    assert_eq!(never.to_vec_bool().unwrap(), vec![false; 4]);
    assert_eq!(always.to_vec_bool().unwrap(), vec![true; 4]);
}

#[test]
fn should_forward_the_permutation_length() {
    let context = gpu_context();
    let key = context.random_key(23).unwrap();

    let permutation = context.random_permutation_arange(8, Some(&key)).unwrap();

    assert_eq!(permutation.shape(), vec![8]);
}
