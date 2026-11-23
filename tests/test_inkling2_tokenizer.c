/* Small synthetic byte-BPE vocabulary for Inkling's GPT4O tokenizer path. */
#include "../ds4.c"
#include <assert.h>

static char *owned_tokens[32];
static size_t owned_token_count;

static void add_piece_token(ds4_vocab *vocab, const char *piece, int token_id) {
    const ds4_str raw = {piece, strlen(piece)};
    uint64_t encoded_length = 0;
    char *encoded = byte_encode(raw, &encoded_length);
    char *owned = xmalloc((size_t)encoded_length + 1u);
    memcpy(owned, encoded, (size_t)encoded_length);
    owned[encoded_length] = '\0';
    free(encoded);
    assert(owned_token_count < sizeof(owned_tokens) / sizeof(owned_tokens[0]));
    owned_tokens[owned_token_count++] = owned;
    vocab->token[token_id] = (ds4_str){owned, encoded_length};
    table_put(&vocab->token_to_id,
              (ds4_str){owned, encoded_length}, token_id);
}

static void expect_tokens(const ds4_vocab *vocab, const char *text,
                          const int *expected, int expected_count) {
    token_vec tokens = {0};
    bpe_tokenize_text(vocab, text, &tokens);
    if (tokens.len != expected_count) {
        fprintf(stderr, "tokenizer test %s: expected %d tokens, got %d\n",
                text, expected_count, tokens.len);
    }
    assert(tokens.len == expected_count);
    for (int i = 0; i < expected_count; i++) {
        if (tokens.v[i] != expected[i]) {
            fprintf(stderr, "tokenizer test %s token %d: expected %d, got %d\n",
                    text, i, expected[i], tokens.v[i]);
        }
        assert(tokens.v[i] == expected[i]);
    }
    token_vec_free(&tokens);
}

int main(void) {
    ds4_vocab vocab = {0};
    table_init(&vocab.token_to_id, 8);
    table_init(&vocab.merge_rank, 1);
    vocab.ignore_merges = true;
    vocab.n_vocab = 200011;
    vocab.token = xcalloc((size_t)vocab.n_vocab, sizeof(vocab.token[0]));
    g_ds4_shape = DS4_SHAPE_INKLING2;

    add_piece_token(&vocab, "foo", 10);
    add_piece_token(&vocab, "BAR", 11);
    add_piece_token(&vocab, " ", 12);
    add_piece_token(&vocab, "123", 13);
    add_piece_token(&vocab, "4", 14);
    add_piece_token(&vocab, "Hello", 15);
    add_piece_token(&vocab, " world", 16);
    add_piece_token(&vocab, "ab", 17);
    add_piece_token(&vocab, "c", 18);
    add_piece_token(&vocab, "HELLO", 19);
    add_piece_token(&vocab, "don't", 20);
    add_piece_token(&vocab, "世界", 21);
    add_piece_token(&vocab, "!/\n/", 22);
    add_piece_token(&vocab, "system", 23);
    add_piece_token(&vocab, " text", 24);
    add_piece_token(&vocab, "Thinking", 25);
    add_piece_token(&vocab, " effort", 26);
    add_piece_token(&vocab, " level", 27);
    add_piece_token(&vocab, ":", 28);
    add_piece_token(&vocab, "0", 30);
    add_piece_token(&vocab, ".", 31);
    add_piece_token(&vocab, "9", 32);
    add_piece_token(&vocab, "hello", 33);
    add_piece_token(&vocab, "1", 34);
    add_piece_token(&vocab, "3", 35);
    add_piece_token(&vocab, "5", 36);
    add_piece_token(&vocab, "Aé", 37);
    add_piece_token(&vocab, "B", 38);
    table_put(&vocab.merge_rank, (ds4_str){"a b", 3}, 0);

    const int split_case[] = {10, 11, 12, 13, 14};
    expect_tokens(&vocab, "fooBAR 1234", split_case,
                  sizeof(split_case) / sizeof(split_case[0]));
    const int space_join_case[] = {15, 16};
    expect_tokens(&vocab, "Hello world", space_join_case,
                  sizeof(space_join_case) / sizeof(space_join_case[0]));
    const int merge_fallback_case[] = {17, 18};
    expect_tokens(&vocab, "abc", merge_fallback_case,
                  sizeof(merge_fallback_case) / sizeof(merge_fallback_case[0]));
    const int uppercase_case[] = {19};
    expect_tokens(&vocab, "HELLO", uppercase_case,
                  sizeof(uppercase_case) / sizeof(uppercase_case[0]));
    const int contraction_case[] = {20};
    expect_tokens(&vocab, "don't", contraction_case,
                  sizeof(contraction_case) / sizeof(contraction_case[0]));
    const int unicode_case[] = {21};
    expect_tokens(&vocab, "世界", unicode_case,
                  sizeof(unicode_case) / sizeof(unicode_case[0]));
    const int unicode_case_boundary[] = {37, 38};
    expect_tokens(&vocab, "AéB", unicode_case_boundary,
                  sizeof(unicode_case_boundary) / sizeof(unicode_case_boundary[0]));
    const int punctuation_case[] = {22};
    expect_tokens(&vocab, "!/\n/", punctuation_case,
                  sizeof(punctuation_case) / sizeof(punctuation_case[0]));

    ds4_engine engine = {0};
    engine.vocab = vocab;
    engine.vocab.system_id = 200002;
    engine.vocab.user_id = 200000;
    engine.vocab.assistant_id = 200001;
    engine.vocab.tool_id = 200003;
    engine.vocab.content_text_id = 200004;
    engine.vocab.content_thinking_id = 200008;
    engine.vocab.end_message_id = 200010;
    engine.vocab.eos_id = 200006;
    engine.vocab.endoftext_id = 199999;
    engine.vocab.bos_id = -1;
    engine.vocab.content_image_id = -1;
    engine.vocab.content_xml_id = -1;
    engine.vocab.content_invoke_tool_json_id = -1;
    engine.vocab.content_invoke_tool_text_id = -1;
    engine.vocab.begin_text_id = -1;

    char effort[64];
    inkling_effort_text(DS4_THINK_NONE, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0"));
    inkling_effort_text(DS4_THINK_MINIMAL, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.1"));
    inkling_effort_text(DS4_THINK_LOW, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.2"));
    inkling_effort_text(DS4_THINK_MEDIUM, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.7"));
    inkling_effort_text(DS4_THINK_HIGH, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.9"));
    inkling_effort_text(DS4_THINK_XHIGH, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.99"));
    inkling_effort_text(DS4_THINK_MAX, effort, sizeof(effort));
    assert(!strcmp(effort, "Thinking effort level: 0.99"));

    ds4_tokens chat = {0};
    ds4_chat_append_message(&engine, &chat, "system", "system text");
    ds4_chat_append_message(&engine, &chat, "user", "hello");
    ds4_chat_append_assistant_prefix(&engine, &chat, DS4_THINK_HIGH);
    assert(chat.len > 10);
    assert(chat.v[0] == engine.vocab.system_id);
    assert(chat.v[1] == engine.vocab.content_text_id);
    assert(chat.v[chat.len - 1] == engine.vocab.assistant_id);
    int effort_message = -1;
    for (int i = 0; i + 2 < chat.len; i++) {
        if (chat.v[i] == engine.vocab.system_id &&
            chat.v[i + 1] == engine.vocab.content_text_id) {
            effort_message = i;
        }
    }
    assert(effort_message > 0);
    assert(chat.v[effort_message - 1] == engine.vocab.end_message_id);
    assert(chat.v[effort_message + 2] == 25);
    assert(chat.v[effort_message + 3] == 26);
    assert(chat.v[effort_message + 4] == 27);
    assert(chat.v[effort_message + 5] == 28);
    assert(chat.v[effort_message + 6] == 12);
    assert(chat.v[effort_message + 7] == 30);
    assert(chat.v[effort_message + 8] == 31);
    assert(chat.v[effort_message + 9] == 32);
    assert(chat.v[effort_message + 10] == engine.vocab.end_message_id);
    ds4_tokens_free(&chat);

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(&engine, "system text", "hello", DS4_THINK_HIGH,
                           &prompt);
    assert(prompt.len > 5);
    assert(prompt.v[prompt.len - 1] == engine.vocab.assistant_id);
    ds4_tokens_free(&prompt);

    ds4_tokens assistant_message = {0};
    ds4_chat_append_message(&engine, &assistant_message, "assistant", "Hello");
    assert(assistant_message.v[assistant_message.len - 5] ==
           engine.vocab.assistant_id);
    assert(assistant_message.v[assistant_message.len - 4] ==
           engine.vocab.content_text_id);
    assert(assistant_message.v[assistant_message.len - 3] == 15);
    assert(assistant_message.v[assistant_message.len - 2] ==
           engine.vocab.end_message_id);
    assert(assistant_message.v[assistant_message.len - 1] == engine.vocab.eos_id);
    ds4_tokens_free(&assistant_message);

    ds4_tokens think_prefix = {0};
    ds4_chat_append_think_prefix(&engine, &think_prefix, DS4_THINK_MINIMAL);
    assert(think_prefix.v[0] == engine.vocab.system_id);
    assert(think_prefix.v[1] == engine.vocab.content_text_id);
    assert(think_prefix.v[2] == 25);
    ds4_tokens_free(&think_prefix);

    ds4_tokens numeric_effort = {0};
    const ds4_think_mode effort_035 =
        (ds4_think_mode)(DS4_THINK_LEVEL_BASE + 35);
    ds4_chat_append_think_prefix(&engine, &numeric_effort, effort_035);
    ds4_chat_append_message(&engine, &numeric_effort, "user", "hello");
    const int expected_numeric_effort[] = {
        engine.vocab.system_id, engine.vocab.content_text_id,
        25, 26, 27, 28, 12, 30, 31, 35, 36,
        engine.vocab.end_message_id,
    };
    assert(numeric_effort.len >=
           (int)(sizeof(expected_numeric_effort) / sizeof(expected_numeric_effort[0])));
    for (size_t i = 0; i < sizeof(expected_numeric_effort) / sizeof(expected_numeric_effort[0]); i++)
        assert(numeric_effort.v[i] == expected_numeric_effort[i]);
    int system_markers = 0;
    for (int i = 0; i < numeric_effort.len; i++)
        if (numeric_effort.v[i] == engine.vocab.system_id) system_markers++;
    assert(system_markers == 1);
    ds4_tokens_free(&numeric_effort);

    ds4_tokens rendered = {0};
    ds4_tokenize_rendered_chat(
        &engine,
        "<|message_model|><|content_text|>Hello<|end_message|>"
        "<|content_model_end_sampling|>",
        &rendered);
    assert(rendered.len == 5);
    assert(rendered.v[0] == engine.vocab.assistant_id);
    assert(rendered.v[1] == engine.vocab.content_text_id);
    assert(rendered.v[2] == 15);
    assert(rendered.v[3] == engine.vocab.end_message_id);
    assert(rendered.v[4] == engine.vocab.eos_id);
    ds4_tokens_free(&rendered);

    table_free(&engine.vocab.token_to_id);
    table_free(&engine.vocab.merge_rank);
    free(engine.vocab.token);
    for (size_t i = 0; i < owned_token_count; i++) free(owned_tokens[i]);
    puts("Inkling2 GPT4O token splitting and direct-word lookup: ok");
    return 0;
}
