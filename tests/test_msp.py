from pi_fpv_companion.fc.msp import (
    HEADER_REQUEST,
    HEADER_RESPONSE,
    MspDecoder,
    encode,
)


def test_encode_empty_payload():
    f = encode(105)  # MSP_RC request
    assert f[:3] == HEADER_REQUEST
    assert f[3] == 0       # size
    assert f[4] == 105     # cmd
    assert f[5] == 105     # checksum = 0 XOR 105 XOR (nothing) = 105
    assert len(f) == 6


def test_encode_with_payload():
    payload = bytes([0x01, 0x02, 0x03])
    f = encode(200, payload)  # MSP_SET_RAW_RC
    assert f[:3] == HEADER_REQUEST
    assert f[3] == 3
    assert f[4] == 200
    assert f[5:8] == payload
    # checksum = 3 ^ 200 ^ 1 ^ 2 ^ 3
    assert f[8] == (3 ^ 200 ^ 1 ^ 2 ^ 3)


def test_decoder_round_trip_single_frame():
    frame = encode(105, b"\x01\x02\x03\x04", direction=HEADER_RESPONSE)
    dec = MspDecoder(accept=HEADER_RESPONSE)
    out = dec.feed(frame)
    assert out == [(105, b"\x01\x02\x03\x04")]


def test_decoder_handles_two_frames_in_one_buffer():
    f1 = encode(101, b"\x10", direction=HEADER_RESPONSE)
    f2 = encode(105, b"\x20\x21", direction=HEADER_RESPONSE)
    dec = MspDecoder(accept=HEADER_RESPONSE)
    out = dec.feed(f1 + f2)
    assert out == [(101, b"\x10"), (105, b"\x20\x21")]


def test_decoder_handles_partial_frame_split_across_feeds():
    frame = encode(105, b"\x01\x02\x03\x04", direction=HEADER_RESPONSE)
    dec = MspDecoder(accept=HEADER_RESPONSE)
    assert dec.feed(frame[:4]) == []
    assert dec.feed(frame[4:]) == [(105, b"\x01\x02\x03\x04")]


def test_decoder_skips_leading_garbage():
    frame = encode(105, b"\x05", direction=HEADER_RESPONSE)
    dec = MspDecoder(accept=HEADER_RESPONSE)
    assert dec.feed(b"junkjunk" + frame) == [(105, b"\x05")]


def test_decoder_resyncs_after_bad_checksum():
    good = encode(105, b"\xAA", direction=HEADER_RESPONSE)
    bad = bytearray(encode(101, b"\xBB", direction=HEADER_RESPONSE))
    bad[-1] ^= 0xFF   # corrupt checksum
    dec = MspDecoder(accept=HEADER_RESPONSE)
    out = dec.feed(bytes(bad) + good)
    # The bad frame is dropped; the good one survives
    assert out == [(105, b"\xAA")]


def test_decoder_rejects_request_direction_when_listening_for_response():
    req = encode(105, b"", direction=HEADER_REQUEST)
    dec = MspDecoder(accept=HEADER_RESPONSE)
    assert dec.feed(req) == []
