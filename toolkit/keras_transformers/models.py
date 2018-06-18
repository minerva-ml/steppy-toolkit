import shutil

from keras.models import load_model
from steppy.base import BaseTransformer

from toolkit.keras_transformers.architectures import vdcnn, scnn, dpcnn, cudnn_gru, cudnn_lstm
from toolkit.keras_transformers.contrib import AttentionWeightedAverage


class KerasModelTransformer(BaseTransformer):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__()
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config

    def reset(self):
        self.model = self._build_model(**self.architecture_config)

    def _compile_model(self, model_params, optimizer_params):
        model = self._build_model(**model_params)
        optimizer = self._build_optimizer(**optimizer_params)
        loss = self._build_loss()
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def _create_callbacks(self, **kwargs):
        raise NotImplementedError

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _build_optimizer(self, **kwargs):
        raise NotImplementedError

    def _build_loss(self, **kwargs):
        raise NotImplementedError

    def persist(self, filepath):
        checkpoint_callback = self.callbacks_config.get('model_checkpoint')
        if checkpoint_callback:
            checkpoint_filepath = checkpoint_callback['filepath']
            shutil.copyfile(checkpoint_filepath, filepath)
        else:
            self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath,
                                custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
        return self


class ClassifierXY(KerasModelTransformer):
    def fit(self, X, y, validation_data, *args, **kwargs):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        self.model.fit(X, y,
                       validation_data=validation_data,
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, X, y=None, validation_data=None, *args, **kwargs):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class ClassifierGenerator(KerasModelTransformer):
    def fit(self, datagen, X, y, datagen_valid=None, X_valid=None, y_valid=None, *args, **kwargs):
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

        fit_args = self.training_config['fit_args']
        flow_args = self.training_config['flow_args']
        batch_size = flow_args['batch_size']
        if X_valid is None:
            self.model.fit_generator(
                datagen.flow(X, y, **flow_args),
                steps_per_epoch=len(X) // batch_size,
                callbacks=self.callbacks,
                **fit_args)
            return self
        else:
            if datagen_valid is None:
                datagen_valid = datagen
            self.model.fit_generator(
                datagen.flow(X, y, **flow_args),
                steps_per_epoch=len(X) // batch_size,
                validation_data=datagen_valid.flow(X_valid, y_valid, **flow_args),
                validation_steps=len(X_valid) // batch_size,
                callbacks=self.callbacks,
                **fit_args)
            return self

    def transform(self, datagen, X, datagen_valid=None, X_valid=None, *args, **kwargs):
        flow_args = self.training_config['flow_args']
        y_proba_train = self.model.predict_generator(
            datagen.flow(X, shuffle=False, **flow_args))
        result = dict(output=y_proba_train)
        if X_valid is not None:
            if datagen_valid is None:
                datagen_valid = datagen
            y_proba_valid = self.model.predict_generator(
                datagen_valid.flow(X_valid, shuffle=False, **flow_args))
            result.update(dict(output_valid=y_proba_valid))
        return result


class PretrainedEmbeddingModel(ClassifierXY):
    def fit(self, X, y, validation_data, embedding_matrix):
        X_valid, y_valid = validation_data
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.architecture_config['model_params']['embedding_matrix'] = embedding_matrix
        self.model = self._compile_model(**self.architecture_config)
        self.model.fit(X, y,
                       validation_data=[X_valid, y_valid],
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, X, y=None, validation_data=None, embedding_matrix=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class CharVDCNNTransformer(ClassifierXY):
    def _build_model(self, embedding_size, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return vdcnn(embedding_size, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first)


class WordSCNNTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return scnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                    filter_nr, kernel_size, repeat_block,
                    dense_size, repeat_dense, output_size, output_activation,
                    max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                    dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                    conv_kernel_reg_l2, conv_bias_reg_l2,
                    dense_kernel_reg_l2, dense_bias_reg_l2,
                    use_prelu, use_batch_norm, batch_norm_first)


class WordDPCNNTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        """
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """
        return dpcnn(embedding_matrix, embedding_size, trainable_embedding, maxlen, max_features,
                     filter_nr, kernel_size, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, conv_dropout, dense_dropout, dropout_mode,
                     conv_kernel_reg_l2, conv_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first)


class WordCuDNNLSTMTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding,
                     maxlen, max_features,
                     unit_nr, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                     rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return cudnn_lstm(embedding_matrix, embedding_size, trainable_embedding,
                          maxlen, max_features,
                          unit_nr, repeat_block,
                          dense_size, repeat_dense, output_size, output_activation,
                          max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                          dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                          rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                          dense_kernel_reg_l2, dense_bias_reg_l2,
                          use_prelu, use_batch_norm, batch_norm_first)


class WordCuDNNGRUTransformer(PretrainedEmbeddingModel):
    def _build_model(self, embedding_matrix, embedding_size, trainable_embedding,
                     maxlen, max_features,
                     unit_nr, repeat_block,
                     dense_size, repeat_dense, output_size, output_activation,
                     max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                     dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                     rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                     dense_kernel_reg_l2, dense_bias_reg_l2,
                     use_prelu, use_batch_norm, batch_norm_first):
        return cudnn_gru(embedding_matrix, embedding_size, trainable_embedding,
                         maxlen, max_features,
                         unit_nr, repeat_block,
                         dense_size, repeat_dense, output_size, output_activation,
                         max_pooling, mean_pooling, weighted_average_attention, concat_mode,
                         dropout_embedding, rnn_dropout, dense_dropout, dropout_mode,
                         rnn_kernel_reg_l2, rnn_recurrent_reg_l2, rnn_bias_reg_l2,
                         dense_kernel_reg_l2, dense_bias_reg_l2,
                         use_prelu, use_batch_norm, batch_norm_first)
