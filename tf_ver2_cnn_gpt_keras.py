import tensorflow as tf

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_out, ker_sz=3):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_out  = d_out
        self.ker_sz = ker_sz
        self.stride = [1, 1, 1, 1]
        self.win_sz = [1, ker_sz, 1, 1]

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(
            ker_sz * d_model / n_heads)
        self.d_heads = int(d_model / n_heads)
        self.d_rsqrt = tf.math.rsqrt(tf.cast(
            ker_sz*self.d_depth, tf.float32))
        
        self.wo = tf.keras.layers.Dense(d_out)
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = [
            batch_size, seq_length, 
            self.n_heads, self.d_depth]
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        x_dims = tf.shape(x)
        batch_size = x_dims[0]
        num_heads  = x_dims[1]
        seq_length = x_dims[2]
        dim_heads  = x_dims[3]

        dim_model  = num_heads * dim_heads
        output_shp = [
            batch_size, seq_length, dim_model]
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def cnn_attn(
        self, x, y, z, mask=None, neg_infty=-1.0e9):
        x_patch = tf.image.extract_patches(
            x, self.win_sz, 
            self.stride, [1, 1, 1, 1], "VALID")
        x_patch = tf.squeeze(x_patch, axis=2)
        
        y_patch = tf.image.extract_patches(
            y, self.win_sz, 
            self.stride, [1, 1, 1, 1], "VALID")
        y_patch = tf.squeeze(y_patch, axis=2)

        z_patch = tf.image.extract_patches(
            z, self.win_sz, 
            self.stride, [1, 1, 1, 1], "VALID")
        z_patch = tf.squeeze(z_patch, axis=2)
        
        x_patch = self.split_heads(x_patch)
        y_patch = self.split_heads(y_patch)
        z_patch = self.split_heads(z_patch)

        # Dot Product Attention. #
        xy_dot_product = tf.matmul(
            x_patch, y_patch, transpose_b=True)
        xy_attn_logits = xy_dot_product * self.d_rsqrt
        
        # Add the mask to the attention mechanism. #
        if mask is not None:
            attn_mask = (mask * neg_infty)
        else:
            attn_mask = tf.zeros([
                tf.shape(x_patch)[1], 
                tf.shape(y_patch)[1]])
        xy_attn_logits += attn_mask

        xy_attn_weights = tf.nn.softmax(
            xy_attn_logits, axis=-1)
        xy_attn_outputs = tf.matmul(
            xy_attn_weights, z_patch)
        return xy_attn_weights, xy_attn_outputs
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        zero_shape = [
            batch_size, self.ker_sz-1, self.d_model]
        x_zero_pad = tf.zeros(
            zero_shape, dtype=tf.float32, name="zero_pad")
        
        # Maintain causality. #
        q_pad = tf.expand_dims(tf.concat(
            [x_zero_pad, self.wq(q)], axis=1), axis=2)
        k_pad = tf.expand_dims(tf.concat(
            [x_zero_pad, self.wk(k)], axis=1), axis=2)
        v_pad = tf.expand_dims(tf.concat(
            [x_zero_pad, self.wv(v)], axis=1), axis=2)
        
        # Dot Product Attention. #
        attn_tuple = self.cnn_attn(
            q_pad, k_pad, v_pad, mask=mask)
        
        attn_wgt = attn_tuple[0]
        attn_out = self.wo(
            self.combine_heads(attn_tuple[1]))
        return attn_out, attn_wgt
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        # Use squared ReLU activation function. #
        return self.ffwd_2(tf.square(self.ffwd_1(x)))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, n_heads, d_model, d_ffwd, 
        d_out, ker_sz=3, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1  = rate1
        self.rate2  = rate2
        self.ker_sz = ker_sz
        self.d_out  = d_out
        
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, self.d_out, ker_sz=ker_sz)
        
        self.lnorm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6)
        self.lnorm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, training=True, mask=None):
        x_embed = x_enc + x_pos
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=mask)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_tuple[0]))
        attn_self_output = self.dropout_1(
            attn_self_output, training=training)
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, n_heads, 
        d_model, d_ffwd, vocab_size, max_seq_length, 
        d_out=None, ker_sz=3, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1  = rate1
        self.rate2  = rate2
        self.ker_sz = ker_sz

        self.n_heads  = n_heads
        self.n_layers = n_layers

        if d_out is None:
            self.d_out = d_model
        else:
            self.d_out = d_out
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for m in range(n_layers):
            tmp_pos_embed.append(
                tf.keras.layers.Embedding(
                    max_seq_length, d_model))
        
        self.dec_embed = tf.keras.layers.Embedding(
            vocab_size, d_model)
        self.pos_embed = tmp_pos_embed
        del tmp_pos_embed
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for m in range(n_layers):
            tmp_dec_layers.append(DecoderLayer(
                n_heads, d_model, d_ffwd, 
                self.d_out, ker_sz, rate1, rate2))
        
        self.dec_layers = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        input_mask = tf.linalg.band_part(
            tf.ones([seq_length, seq_length]), -1, 0)
        input_mask = 1.0 - input_mask
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, 
                training=training, mask=input_mask)
            layer_input  = layer_output
        return layer_output

class GPTDecoder(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, 
        d_model, d_ffwd, vocab_size, max_seq_length, 
        d_out=None, ker_sz=3, rate1=0.1, rate2=0.1):
        super(GPTDecoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1  = rate1
        self.rate2  = rate2
        self.ker_sz = ker_sz

        self.n_heads  = n_heads
        self.n_layers = n_layers

        if d_out is None:
            self.d_out = d_model
        else:
            self.d_out = d_out
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.vocab_size = vocab_size
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, n_heads, d_model, d_ffwd, 
            vocab_size, max_seq_length, self.d_out, 
            ker_sz=ker_sz, rate1=rate1, rate2=rate2)
        self.p_decoder = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=True):
        dec_outputs = self.gpt_model(
            x, training=training)
        dec_logits  = self.p_decoder(dec_outputs)
        return dec_logits
    
    def infer(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_logits = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)