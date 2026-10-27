// Current route (tab, frame, browse filter) shared by the modules; builds hashes that keep all three.
export const nav = { tab: "edit", stem: "", qs: "" };
export const hashFor = (stem = nav.stem, tab = nav.tab, qs = nav.qs) => `#${tab}${stem ? "/" + encodeURIComponent(stem) : ""}${qs ? "?" + qs : ""}`;
export const replaceHash = (h) => { if (location.hash !== h) history.replaceState(null, "", h); };
