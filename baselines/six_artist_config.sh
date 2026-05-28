#!/bin/bash
# Shared six-artist benchmark config for future concurrent erasure experiments.

set -euo pipefail

SPACE_SIX_ARTISTS=(
  "Kelly McKernan"
  "Van Gogh"
  "Tyler Edlin"
  "Thomas Kinkade"
  "Kilian Eng"
  "Ajin: Demi Human"
)

SPACE_SIX_ARTISTS_SLUGS=(
  "Kelly_McKernan"
  "Van_Gogh"
  "Tyler_Edlin"
  "Thomas_Kinkade"
  "Kilian_Eng"
  "Ajin_Demi_Human"
)

SPACE_SIX_ARTIST_UCE_CONCEPTS="${SPACE_SIX_ARTIST_UCE_CONCEPTS:-Kelly McKernan; Van Gogh; Tyler Edlin; Thomas Kinkade; Kilian Eng; Ajin: Demi Human}"
SPACE_SIX_ARTIST_GUIDE_CONCEPTS="${SPACE_SIX_ARTIST_GUIDE_CONCEPTS:-art; art; art; art; art; art}"
SPACE_DEFAULT_PRESERVE_CONCEPTS="${SPACE_DEFAULT_PRESERVE_CONCEPTS:-Monet; Rembrandt; Warhol; Picasso; Claude Monet; John Singer Sargent}"
SPACE_SIX_ARTIST_EXP_NAME="${SPACE_SIX_ARTIST_EXP_NAME:-uce-six-artists-concurrent}"
