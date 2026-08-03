"""Reusable branch and terminal proof-source selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal

from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import DirectionMark, SmilesPolicy
from grimace._south_star1.prepared_runtime import SouthStarPreparedMol, SouthStarRuntimeOptions
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_branch_transition_artifact import (
    writer_branch_transition_artifact_for_support,
)
from grimace._south_star1.writer_events import WriterRingEndpointEmitted, WriterRingEndpointPaired
from grimace._south_star1.writer_snapshot import (
    WriterDecoderBoundary,
    capture_writer_frontier_snapshot,
)
from tests.south_star1.writer_test_context import (
    initial_writer_snapshot,
    prepare_writer_facts,
    writer_runtime_options,
)
from tests.south_star1.writer_test_fixtures import shared_directional_ring_carrier_facts

SharedRingBranchPhase = Literal["opening", "pair"]


@dataclass(frozen=True, slots=True)
class SharedRingBranchSourceAddress:
    phase: SharedRingBranchPhase
    direction_mark: DirectionMark
    predecessor_branch_certificate_digests: tuple[str, ...]
    source_cursor_digest: str
    target_branch_certificate_digest: str
    target_emitted_text: str
    target_successor_cursor_digest: str
    expected_branch_artifact_digest: str


def _address(
    phase: SharedRingBranchPhase,
    direction_mark: DirectionMark,
    predecessor_branch_certificate_digests: tuple[str, ...],
    source_cursor_digest: str,
    target_branch_certificate_digest: str,
    target_emitted_text: str,
    target_successor_cursor_digest: str,
    expected_branch_artifact_digest: str,
) -> SharedRingBranchSourceAddress:
    return SharedRingBranchSourceAddress(
        phase,
        direction_mark,
        predecessor_branch_certificate_digests,
        source_cursor_digest,
        target_branch_certificate_digest,
        target_emitted_text,
        target_successor_cursor_digest,
        expected_branch_artifact_digest,
    )


SHARED_RING_BRANCH_SOURCE_ADDRESSES = (
    _address(
        "opening", DirectionMark.REV,
        ("2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",),
        "4412694361a0196708ff60c818e24de7f6e620f044362bc4c7e97842a89f7053",
        "00fb8cc5cfd284744f7d268333dd050a65cfae4b29ae975eade2b0820c35531c",
        "\\1",
        "18c113f30b73800efe59988c2e32e804ede29fb89774d675158b1d3e9eced627",
        "1d1c8209bb88935d1774ce5e47dd5bedea7746435ed6bf504e6b50390e5c7df1",
    ),
    _address(
        "opening", DirectionMark.ABSENT,
        ("2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",),
        "4412694361a0196708ff60c818e24de7f6e620f044362bc4c7e97842a89f7053",
        "b49262a55506f6fd49d0380115bdfa98bf74a2b6fc129754c89a6a5ee8d90930",
        "1",
        "50d97cf2f4fb172c7c00b5e888b359eb8a2653ccceb95071181285de0511ffcf",
        "a6cbac10a2058e5df696814b467117b0b76f861586d190e683c88c91cbc63479",
    ),
    _address(
        "opening", DirectionMark.FWD,
        ("2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",),
        "4412694361a0196708ff60c818e24de7f6e620f044362bc4c7e97842a89f7053",
        "f5355f7bc7788c645313f8f00059b8fc0d7b6e32dd739520d2f2999f94298e48",
        "/1",
        "81689a5fb5f5829a1dd822762b7ec8ab7b47a7c69753599474c0991d97f917d8",
        "d385f851f830d7b654a967ffcf378452dd8fb7695490617df70b4ecdeb95e852",
    ),
    _address(
        "pair", DirectionMark.REV,
        (
            "2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",
            "00fb8cc5cfd284744f7d268333dd050a65cfae4b29ae975eade2b0820c35531c",
            "8e3205d8fd241aabd49332dbeaceed54964e8b88e99b34d5b0ef669a5f6fc483",
            "9c7fff04b4bba3def880384c9c01ee36523c06614fe65777bcbc41856917a90b",
            "821b897b89f898b50b9d8a00eb656305ffb09d169f862935f8d4f0beb8828565",
            "89a086448ea65699f423574874a84513d407beebaa07c12661b17de0db74614a",
            "0553f8ab8a5e39750129f0a7db3e19af6c57633d50df85e57c8a705bde61b076",
            "60311edba3f4bec48b7f2cd78e9f60eecacaaeb6501d7fe5fa2b4460c411ff85",
            "42737fe98f0d8e7704b0afb7cdfc9b371c48b43380e6b06036d5acd4016aefe5",
            "18f53c0147c46742342d5b2d0dbcd9b0587b74f0a2c22475a9f8137f1551c924",
            "d63587fadcff8ab3c5f2b23bda4000fdfa797ea422445d0240d31027b419a11a",
            "7795e10b5463201ebd1e8f5773ca3316ec244703dd3a22e11296b342dbb53cb3",
            "3d16d86f5bff74e464000b320653660c0925e2188e5be3c37f9422adf4b6c0d7",
            "d8c28cafe7c49e3c19b2b2a2ec3229c6757204763dbe484ff5789d13479e41e2",
            "7d7bb91d5a3470e21c550cf516d20aa44a26dbf678d2af92575e1349ee29a7c9",
            "b6447a14910df9d2ac8b5b7da9ee218f2f01b67e88599531e1bce3bf9dd88042",
            "4b97d122eabc344eb657b28200e318bdb6d930f635ef6fbf3e49cfd838ab3e6f",
            "fb093ad1635529096aaf89c0ca99998cd70fe1f2067262797c6b48de06434cb7",
            "f56e6ae3e2bfa4ed522b3bf45ef96242d0b6f739e5a1ef9275dfacec5b8373db",
            "5c401fe9ae156023bc2acef36190b18161eac89f6fd158a7e0f3c5e0c63cf79e",
            "ae4ed72124a6f1c9c67a23f335e28455f864fb7558fbcd139c4796e36b61bfd5",
            "8da955ed77dcf26a4bdf1166d3aabd0316b6ec2ba92ccd5bf71ec238196c13f6",
            "ec12ea30419587dc1731ff02b6a3073007741fcdff232165bd12f96c7405d9cb",
        ),
        "fa4ccffa459859cb09152d8bad8a1f2de466e937015f7f067d5f899fbc27f800",
        "6bf9933a024a55f26f6cb6af129da666d4e6e22b81e443ee3f5a46600a8e5ef5",
        "1",
        "732983532056851b1cf974325877c71d3445f5b13704526b4696b88f160bfc18",
        "a43a15f1f77a0e9cf4492d2b8ddae1f4d44ff858bddae14d6405d3c5435f6b19",
    ),
    _address(
        "pair", DirectionMark.ABSENT,
        (
            "2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",
            "b49262a55506f6fd49d0380115bdfa98bf74a2b6fc129754c89a6a5ee8d90930",
            "bb93910e4daf551cf99b55fe1c17d4ef3f609215ba0b4bae7a948b69c791b6bd",
            "ca92d42a435a52018d0cb8d21be928c76f22680f4bfbafdbf3b81ab96e39c0ce",
            "3fb380fc317a89dd75279bd56b4ebda4aa5e7c409f4a2fa7bcc9b6192d4ae8d8",
            "8b3765b23bafa4a7da895ea4defd09f93100cf332699e758277c4f4d33582f5e",
            "fb6e6149cb02c1d07634bd5d6ad233c5b0108f54fc6f6b47e59bf6972227e502",
            "d7551314319ae0f008053884405fc23ce690143447740c9577bbc3ecb960d51d",
            "cb3f1c8a274d2cd5de516f4cda98b6ac4b4ed2a0ea0d25c8274f09d38daeb850",
            "bc864d478d891b70d21419666066ce68ebee48da7a756e2355990c060d215dbf",
            "e032220b057d789fce1d1c07a2acb906146f1229a77e3f25f9d73474a4d02531",
            "20b4f50bb010663fa1a13dda5a117eff19ad647650fbc6b32ead7ce77c21bfa7",
            "3d3115ed7c5fb79087e826b8e630bd875f8d02be0f863ec874614bcfefc97223",
            "55f1d0fb180e4f4781cb6b1dfd95bef7c5328bca8196936097ade743e53e7852",
            "9f3adcb56098a7454a2a7515966381dc33c1f73dd1f78a38ec73b55549283a86",
            "b77155af0f7dfaab2db01f35481e99178fd1f2e76b13deabd191109a617daa16",
            "734aef45f94494533d16a5473aa54ccbc126d98e297da6355d85ab6cb55dbd5c",
            "5de81cc8a1c0d676594528c501814f6b285c8b1db6ba0631cbe68aac05693eb7",
            "a3a2274b40bc2c1ca83b3030c47782d018cb7b370a7daf59a62b88f132b11078",
            "b8ef7d1554e6b1376bc0944c2cbb7a7529dfb6512c1d8ecbee049c4451fa21e5",
            "3cc01170070936f74cc9aa4329195205e24591fdfcb3fff20b19fdd1b5d3b02a",
            "15627929802ff5839859b76bb9360d8fbdc0cbb2556fcbfa1e654b60a987a429",
            "4ebc07753b7f9900d0cd9a35c6e38f751bb6129c746c954be8564c50909876ad",
        ),
        "01c6d725cb03f5e3df6c4db223199a75802c0505e107584e5988f9ae2f84ee71",
        "d6f7395df681523328d45c5542ee0c350509e4b834e5954566fe4b714d2bc423",
        "1",
        "127c5c2dd2674146168d939ecc7c33940856af7850659d97651e3e7296b55cc4",
        "926e92ae20ad6d970f8a84c1eea08653b8abb652a0a1cf983f3996015ff4354f",
    ),
    _address(
        "pair", DirectionMark.FWD,
        (
            "2a9492bee38ce814552c1c99b975e60bf307e5c0f3bb609f190221d837c07a90",
            "f5355f7bc7788c645313f8f00059b8fc0d7b6e32dd739520d2f2999f94298e48",
            "d8b014dcd5f6235ef1de64602e8d2292fb964449146c6a1bb14d7c51bd87859f",
            "1ea54f1793be6d35162ec0c35e327a557a1cfe6495a1b589e9928722054efe29",
            "9fb22b7b3c75c6aa60345c616f48cbc547bbebe4bc0fc528b28a0e0cc2e86521",
            "871a7895a162aadffe5360e364fb16ca56f68a5d670d83b5cfc05968a195f4ed",
            "80e4e580d336b5c4842d531d1cb764d8fb00866c6fe88c0a20352c309f0ff8c6",
            "311c6f3db274c9b949873e0a4f31a7623859c879f70983f36540ca2014aaafb4",
            "275bcb1238922cb09510b15a1cb5cc75adf41f14e9f356a633f1a7d7b5a15bd4",
            "96c1be34e90a9a11e140c7aff11c4fc90fff012e014585a0934bb27cdf4cfee7",
            "a73d10f8377b0aa9cc089934566ca1a118d463f55a7f09b8cc7ba6a6494aaefd",
            "86a7a6ce95f2c31331987baea7789555f81511a43cbc39b23de2b5114a63d300",
            "d7fe5f451673bf2897afa08c375beeed4a25e525c8641e41c22f21fa3cb8bd00",
            "46656e7a7254d892ee0ea347993880e8016e0d28dd9be5fa90cf9fa7ca9a9368",
            "35c4d26656013ed99da6515fe89378c4fbeb19aabff9f6f0e7b9a76476da9d58",
            "ead04b98ae653756cf0316c6b4b89d29f368a4552d0aaeee68676c0a13977dc3",
            "f0086bd5c98ce25a3d99354f05c33299508c09ba893862e8d261176dc948a08d",
            "e42fe21fc06e7617368bf56b38ba2d76d3e3b6d6d1bb2eb7a9f47192d465ad8f",
            "2cfb87e31b3ed79a9c7a7613c155b964eb47756d815f67475c0c562a4c670068",
            "632a0c13b85f3d3aabc4ff01aede4001a336711e0293031a981c682e4bcd6f5b",
            "37a804d9c8a50a817de8bf9e9f6a608c71511c1a018475e18286e172a172a135",
            "71fac20cf53d64983c4f2512ecedc1a178f15609b35c122019f90f4392a62dcb",
            "c348dd1f09bba937c4a5f120890d58f6d56f53f17f8a46b8c8c94a889b73b97a",
        ),
        "263439510b24e51435fbd168ef8aee188cdd9fe80f04caaf6bce81418fad6f22",
        "810ccecbf0823bd3110b50257fae3ec31ba211002ede2bc2cfd94c3166b826cc",
        "1",
        "7d1a62c3e3d1c2d1f761b360d253deb6bb44427d2706299923cbefd96599a34f",
        "e6da221414c2b4d01c9079dc0bf1ff78faf4b28f27acfbd3d7689055da28bb06",
    ),
)


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def validate_shared_ring_branch_source_addresses(
    addresses=SHARED_RING_BRANCH_SOURCE_ADDRESSES,
) -> None:
    expected_keys = tuple(
        (phase, mark)
        for phase in ("opening", "pair")
        for mark in sorted(DirectionMark, key=lambda item: item.value)
    )
    actual_keys = tuple((address.phase, address.direction_mark) for address in addresses)
    if actual_keys != expected_keys:
        raise ValueError("shared-ring source address order or key matrix is invalid")
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("shared-ring source address keys must be unique")
    target_digests = set()
    for address in addresses:
        digests = address.predecessor_branch_certificate_digests
        if not digests:
            raise ValueError("shared-ring source address path cannot be empty")
        for digest in (*digests, address.source_cursor_digest,
                       address.target_branch_certificate_digest,
                       address.target_successor_cursor_digest,
                       address.expected_branch_artifact_digest):
            if not _HEX_DIGEST.fullmatch(digest):
                raise ValueError("shared-ring source address digest is not canonical")
        if not address.target_emitted_text:
            raise ValueError("shared-ring target emitted text cannot be empty")
        if address.target_branch_certificate_digest in target_digests:
            raise ValueError("shared-ring target branch certificates must be unique")
        target_digests.add(address.target_branch_certificate_digest)


@dataclass(frozen=True, slots=True)
class WriterBranchProofSource:
    phase: SharedRingBranchPhase
    direction_mark: DirectionMark
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    snapshot: object
    support: object


@dataclass(frozen=True, slots=True)
class WriterTerminalProofSource:
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    snapshot: object
    support: object
    policy: SmilesPolicy | None


@lru_cache(maxsize=1)
def shared_ring_branch_sources() -> tuple[WriterBranchProofSource, ...]:
    facts = shared_directional_ring_carrier_facts()
    options = writer_runtime_options(rooted_at_atom=1)
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    found = {}
    while pending and len(found) < 6:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        snapshot = (
            initial_writer_snapshot(prepared, options)
            if depth == 0
            else capture_writer_frontier_snapshot(
                prepared=prepared,
                runtime_options=options,
                cursor=cursor,
                decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
            )
        )
        for support in batch.supports:
            for event in support.events:
                if isinstance(event, WriterRingEndpointEmitted) and event.bond == BondId(1):
                    found.setdefault(("opening", event.direction_mark), (facts, options, prepared, snapshot, support))
                if isinstance(event, WriterRingEndpointPaired) and event.bond == BondId(1):
                    found.setdefault(("pair", event.first_endpoint_direction_mark), (facts, options, prepared, snapshot, support))
            pending.append((support.successor_cursor, depth + 1))
    if len(found) != 6:
        raise AssertionError(f"missing shared-ring branch sources: {sorted(found)}")
    return tuple(
        WriterBranchProofSource(phase, mark, *found[(phase, mark)])
        for phase in ("opening", "pair")
        for mark in sorted(DirectionMark, key=lambda item: item.value)
    )


def shared_ring_branch_source(
    phase: SharedRingBranchPhase,
    direction_mark: DirectionMark,
) -> WriterBranchProofSource:
    for source in shared_ring_branch_sources():
        if source.phase == phase and source.direction_mark is direction_mark:
            return source
    raise ValueError(f"unknown shared-ring branch source: {phase!r}, {direction_mark!r}")


def first_terminal_proof_source(
    facts: MoleculeFacts,
    runtime_options: SouthStarRuntimeOptions,
    *,
    policy: SmilesPolicy | None = None,
) -> WriterTerminalProofSource:
    prepared = prepare_writer_facts(facts, policy=policy)
    snapshot = initial_writer_snapshot(prepared, runtime_options)
    from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot

    for depth in range(256):
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        if batch.terminal_supports:
            return WriterTerminalProofSource(
                facts, runtime_options, prepared, snapshot, batch.terminal_supports[0], policy
            )
        support = batch.supports[0]
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=runtime_options,
            cursor=support.successor_cursor,
            decoder_boundary=WriterDecoderBoundary(depth + 1),
        )
    raise AssertionError("terminal support not reached")


__all__ = (
    "SHARED_RING_BRANCH_SOURCE_ADDRESSES",
    "SharedRingBranchSourceAddress",
    "WriterBranchProofSource",
    "WriterTerminalProofSource",
    "first_terminal_proof_source",
    "shared_ring_branch_source",
    "shared_ring_branch_sources",
    "validate_shared_ring_branch_source_addresses",
)
