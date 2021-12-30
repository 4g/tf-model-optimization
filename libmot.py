import tempfile
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import gzip

class ModelOptimizers:
    @staticmethod
    def callbacks(log_dir):
        return []

    @staticmethod
    def modify(model):
        return model

    @staticmethod
    def save(model):
        return model


class TFlite:
    @staticmethod
    def save(model, fname, quantize=False):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()

        with gzip.open(fname, 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)

class Prune(ModelOptimizers):
    @staticmethod
    def callbacks(log_dir):
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
        ]
        return callbacks

    @staticmethod
    def modify(model):
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
        }
        model = prune_low_magnitude(model, **pruning_params)
        model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=["accuracy"])
        return model

    @staticmethod
    def save(model):
        model = tfmot.sparsity.keras.strip_pruning(model)
        return model

    @staticmethod
    def debug(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                if "kernel" not in weight.name or "centroid" in weight.name:
                    continue
                weight_size = weight.numpy().size
                zero_num = np.count_nonzero(weight == 0)
                print(
                    f"{weight.name}: {zero_num / weight_size:.2%} sparsity ",
                    f"({zero_num}/{weight_size})",
                )

class QAT(ModelOptimizers):
    @staticmethod
    def modify(model):
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        q_aware_model.compile(loss="categorical_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(0.001),
                 metrics=["accuracy"])
        return q_aware_model

class WeightCluster(ModelOptimizers):
    @staticmethod
    def modify(model):
        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        clustering_params = {
            'number_of_clusters': 16,
            'cluster_centroids_init': CentroidInitialization.LINEAR
        }
        clustered_model = cluster_weights(model, **clustering_params)
        clustered_model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=["accuracy"])
        return clustered_model

    @staticmethod
    def save(model):
        return tfmot.clustering.keras.strip_clustering(model)

    @staticmethod
    def debug(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Wrapper):
                weights = layer.trainable_weights
            else:
                weights = layer.weights
            for weight in weights:
                # ignore auxiliary quantization weights
                if "quantize_layer" in weight.name:
                    continue
                if "kernel" in weight.name:
                    unique_count = len(np.unique(weight))
                    print(
                        f"{layer.name}/{weight.name}: {unique_count} clusters "
                    )