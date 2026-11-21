// A picture of a project's video (a scanned frame, a crop of one, a preview). mmPicture and mmPublic are what the
// developer hooks' /grab reads: it greys every picture whose project isn't public domain (AGENTS.md, Privacy).
//
// Every scaled-down Image in the app sets sourceSize to the size it is drawn at (x Screen.devicePixelRatio): Qt then
// decodes it at that size with its smooth (area-averaging) scaler, and the GPU draws it about 1:1. Left to the GPU,
// a big reduction (a 4K frame in a 300 px crop, the 824 px icon in 168 px) skips most source pixels and looks jagged;
// `mipmap: true` didn't help here.
import QtQuick

//
// A crop of a frame is cut and scaled by the project's server (decide.rs crop_url): sized() adds the size to ask for,
// the next of a few fixed steps above the size it's drawn at (so a resize doesn't refetch at every pixel, and the
// GPU never reduces by more than 1.5).
Item {
    property bool mmPicture: true
    property bool mmPublic: false

    function px(n) {
        const steps = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
        for (const s of steps) if (s >= n) return s
        return steps[steps.length - 1]
    }
    function sized(url, n) { return url ? url + "&px=" + px(n) : "" }
}
