# Market and platform notes (researched 2026-11-23; verify before relying on any figure)

## Platform risk: Meta Wearables Device Access Toolkit (checked 2026-11-23 against developers.meta.com/wearables/faq)
- Status: DAT 1.0 is a stable release after a year of developer preview, rolling out from 2026-09-30 (FAQ).
- Publishing is STILL not open. The FAQ says developers will "soon" be able to submit DAT and Web App experiences, with timing to come.
  Builds can go to limited audiences through release channels (Wearables Developer Center), not the public. Select partners only for public publishing
  (Meta developer-preview blog, 2025). Late-Sept 2026 reports (vr.org) confirm Meta missed its "GA publishing in 2026" aim so far; the FAQ checked today still says "soon".
- Implication: a commercial version can today reach only testers inside your own organisation (or Meta partners). Re-check the FAQ before planning any sale.
- Terms: the FAQ links Wearables Developer Terms and an Acceptable Use Policy. Read both before commercial use (not reviewed here).
- Hardware: supported glasses are Ray-Ban Meta Gen 1/2, Meta Ray-Ban Display, Oakley Meta HSTN and Vanguard; capability varies by device.
  Mock Device Kit does not support Display glasses. Capabilities listed for 1.0: camera, microphone, audio, motion/IMU.
- Markets: only users in Meta AI glasses supported countries get full toolkit access, including the Developer Center.
- Meta AI Connectors and Web Apps are separate paths (Web Apps are Display-glasses only); this project uses neither.

## Competitors (prices vary by source and date; treat as ranges)
- Encircle: field documentation for restoration; ~$250-$270/month small-shop plan with unlimited users (SourceForge, GetApp, Capterra). Large Shop ~$650/month (GetApp).
- magicplan: ~$9.99/user/month entry on one aggregator; Estimate tier $74.99-$89.99/user/month (Capterra UK). Sources disagree.
- CompanyCam: Capterra lists Pro $99, Premium $149, Elite $249 per user/month; other pages list $27-$63. Sources disagree.
- Xactimate: estimating platform, not a capture app; a comparison page lists $149-$250+/seat (that page is a competitor's marketing, low trust).
- Gap those tools do not obviously fill: hands-free capture on glasses with voice commands. Unproven by this project.

## Terms of the take-home
The brief text provided contains no IP-ownership clause. That is not the same as "you own it". Ask the company in writing
(after the 72-hour window, since they will not answer questions during it) before selling or open-sourcing anything built for it.
