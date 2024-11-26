from vllm.sequence import SequenceData, SequenceGroupMetadata, ExecuteModelRequest, SequenceGroupState
from vllm.sampling_params import SamplingParams


def decoding_execute_model_req(msgspec_emq):
    assert len(msgspec_emq) == 12, 'Wrong length of ExecuteModelRequest'

    seq_group_metadata_list_raw = msgspec_emq[0]
    blocks_to_swap_in = msgspec_emq[1]
    blocks_to_swap_out = msgspec_emq[2]
    blocks_to_copy = msgspec_emq[3]
    virtual_engine = msgspec_emq[4]
    num_lookahead_slots = msgspec_emq[5]
    running_queue_size = msgspec_emq[6]
    previous_hidden_states = msgspec_emq[7]
    num_steps = msgspec_emq[8]
    finished_requests_ids = msgspec_emq[9]
    last_sampled_token_ids = msgspec_emq[10]
    async_callback = msgspec_emq[11]

    seq_group_metadata_list: List[Union[SequenceGroupMetadata]] = []
    for raw_metadata in seq_group_metadata_list_raw:
        if raw_metadata[0] == 'SequenceGroupMetadata':
            request_id = raw_metadata[1]
            is_prompt = raw_metadata[2]

            seq_data_raw = raw_metadata[3]
            seq_data: Dict[int, SequenceData] = {}
            for key, value in seq_data_raw.items():
                key = int(key)
                seq_data[key] = SequenceData(
                    _prompt_token_ids=value.get('_prompt_token_ids_tuple', ()),
                    _output_token_ids=value.get('_output_token_ids', ()),
                    _cumulative_logprob=float('inf'),
                    _num_computed_tokens=value.get('_num_computed_tokens', 0)
                )

            sampling_params_raw = raw_metadata[4]
            sampling_params = SamplingParams(
                n=sampling_params_raw.get('n', 1),
                presence_penalty=sampling_params_raw.get('presence_penalty', 0.0),
                frequency_penalty=sampling_params_raw.get('frequency_penalty', 0.0),
                repetition_penalty=sampling_params_raw.get('repetition_penalty', 1.0),
                temperature=sampling_params_raw.get('temperature', 1.0),
                top_p=sampling_params_raw.get('top_p', 1.0),
                top_k=sampling_params_raw.get('top_k', -1),
                max_tokens=sampling_params_raw.get('max_tokens', 16),
                min_tokens=sampling_params_raw.get('min_tokens', 0),
                stop=sampling_params_raw.get('stop', []),
                stop_token_ids=sampling_params_raw.get('stop_token_ids', []),
                ignore_eos=sampling_params_raw.get('ignore_eos', False),
                logprobs=sampling_params_raw.get('logprobs', None),
                prompt_logprobs=sampling_params_raw.get('prompt_logprobs', None),
                skip_special_tokens=sampling_params_raw.get('skip_special_tokens', True),
                spaces_between_special_tokens=sampling_params_raw.get('spaces_between_special_tokens', True)
            )

            block_tables_raw = raw_metadata[5]
            block_tables: Dict[int, List[int]] = {int(k): v for k, v in block_tables_raw.items()}

            state = SequenceGroupState(
                num_steps=num_steps
            )

    seq_group_metadata_list.append(SequenceGroupMetadata(
        request_id=request_id,
        is_prompt=is_prompt,
        seq_data=seq_data,
        sampling_params=sampling_params,
        block_tables=block_tables,
        do_sample=raw_metadata[6],
        token_chunk_size=raw_metadata[7],
        lora_request=raw_metadata[8],
        computed_block_nums=raw_metadata[9],
        state=state,
        multi_modal_data=raw_metadata[11],
        mm_processor_kwargs=raw_metadata[12],
        encoder_seq_data=raw_metadata[13],
        cross_block_table=raw_metadata[14],
        prompt_adapter_request=raw_metadata[15]
    ))

    return ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        virtual_engine=virtual_engine,
        num_lookahead_slots=num_lookahead_slots,
        running_queue_size=running_queue_size,
        previous_hidden_states=previous_hidden_states,
        num_steps=num_steps,
        finished_requests_ids=finished_requests_ids,
        last_sampled_token_ids=last_sampled_token_ids,
        async_callback=async_callback
    )
