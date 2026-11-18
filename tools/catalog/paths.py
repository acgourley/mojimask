"""The screen catalog's paths (run.py runs them): each a function of ui, a person's way through the app, on a fresh copy
of the jazz project. ui.click(key) / ui.wait(key) find things by their strings.js key; ui.shot(name) captures the
screen; ui.faces_answered() answers what a path only needs done, through the app's own API.

Keep a path to what a person does and sees: no reaching into the pages' code, so a path breaks only when the app's
way through changes (and then it should). Getting to where a path starts, when another path already shows it, goes in
`with ui.quietly("note"):` (done and tested, not captured). Add a path for each new flow worth seeing; register it in
PATHS, in the order people meet them.
"""


def path(title):
    def mark(f):
        f.title = title
        return f
    return mark


@path("A new project: the result, then Refine")
def first_open(ui):
    """A project just scanned: it opens on the finished video with its styles, then Refine goes to its questions,
    the is-it-a-face ones first."""
    ui.goto()
    ui.wait("projects.title")
    ui.shot("projects", "The Projects list")
    ui.open_project()
    ui.wait("result.head", timeout=60000)
    ui.wait("result.refine")
    ui.shot("result", "The finished video, before any answers")
    ui.click("result.refine")
    ui.wait("nav.decisions")
    ui.shot("decisions.first", "The first question", wait=1500)


@path("A same-person question, answered No, then found in Answered")
def same_no(ui):
    """With every is-it-a-face question answered, Decisions opens on the who's-who questions: answer No, see the next
    one open, then find the answer again under Answered (its row's short answer, and the card with No chosen)."""
    ui.faces_answered()
    with ui.quietly("Opened the project and chose Refine the covers (shown in \u201cA new project\u201d)"):
        ui.open_decisions()
    ui.wait("decisions.q.same")
    ui.shot("decisions.same.open", "An open same-person question")
    ui.click("decisions.a.different_people")
    ui.shot("decisions.same.after_no", "After No: the next question opens", wait=1500)
    ui.click("decisions.filter.answered")
    ui.click("decisions.short.same")
    ui.shot("decisions.answered", "Your No, found again under Answered: the row's short answer and the card with No chosen", wait=800)


@path("A same-person question, answered Yes, then People")
def same_yes(ui):
    """Answer Yes, then see the joined person on the People step."""
    ui.faces_answered()
    with ui.quietly("Opened the project, chose Refine the covers, and got the same-person question shown in \u201cA same-person question, answered No\u201d"):
        ui.open_decisions()
        ui.wait("decisions.q.same")
    ui.click("decisions.a.same_person")
    ui.shot("decisions.same.after_yes", "After Yes", wait=1500)
    ui.click("nav.people")
    ui.wait("people.their_faces")
    ui.shot("people", "The People step: a card per person, the joined one counting its parts", wait=1000)


@path("The scan's deciding step (a replay)")
def scan_replay(ui):
    """The scan screen's "Deciding who's who" step, replayed from the project's last scan at 30x: the pair-by-pair
    stage while it's on, and the questions at the end."""
    with ui.quietly("Opened the app on a replay of the project's last scan, at 30x"):
        ui.goto(f"vizreplay={ui.project}&x=30")
    ui.wait("viz.pairs.title", timeout=90000)
    ui.shot("scan.pairs", "Comparing the pieces, pair by pair (mid-animation: the counts are still rising)")
    ui.wait("viz.same.sub", timeout=90000)
    ui.shot("scan.questions", "The end of the step: the questions left for you", wait=1500)


@path("A box in the editor, its questions (R), and the answers rules gave")
def editor_rules(ui):
    """A same-person question's green box opened in the editor (click its picture), then R: the questions about that
    box, where the ones a rule settled say so ("· rule"), the rule as the tooltip. Jazz, set up so these two boxes
    show every same-person rule: box 439 (asked next; No to this person's 4 closest; different depths; and, after
    your Yes, kept apart by another answer), and its grey box, 287 (a weak match, never asked)."""
    ui.faces_answered()
    no = "different people"
    ui.answered({"same:5:112": no, "same:5:103": no, "same:5:136": no, "same:5:129": no, "same:7:287": no,
                 "same:30:411": no, "same:15:381": no},
                "Answered No to 7 same-person questions: 4 about one person (so their weaker matches aren't asked), and the ones asked before box 439's")
    with ui.quietly("Opened the project and chose Refine the covers (shown in \u201cA new project\u201d)"):
        ui.open_decisions()
        ui.wait("decisions.q.same")
    ui.shot("decisions.same.439", "The same-person question about box 439 (green) and box 287 (grey)")
    ui.click("decisions.pic.open_tip", tip=True, within=ui.words("decisions.pic.green"))
    ui.wait("edit.sel.their_face", timeout=60000)
    ui.shot("edit.selected", "Box 439, selected in the editor at that moment: its panel, with the questions about it and who answered", wait=1500)
    ui.press("r", on="edit.sel.their_face")
    ui.wait("decisions.piece.all")
    ui.shot("decisions.piece", "R: the questions about box 439, the open one first, then the ones settled for you", wait=1000)
    ui.click("decisions.a.same_person")
    ui.shot("decisions.piece.after_yes", "After Yes: another question about box 439 is now settled by your answers (kept apart)", wait=1000)
    rule = lambda k: ui.words(k) + " \u00b7 " + ui.words("decisions.rule")
    ui.click("decisions.sa.kept_apart + decisions.rule", text=rule("decisions.sa.kept_apart"))
    ui.shot("decisions.rule.kept_apart", "Kept apart by your other answers: no button for it, so a line under the buttons; the rule is its tooltip")
    for i in range(2):
        ui.click("decisions.sa.no + decisions.rule", text=rule("decisions.sa.no"), nth=i)
        ui.shot(f"decisions.rule.no{i + 1}", "A rule's No: the No button marked \u201crule\u201d; the rule is its tooltip")
    ui.click("decisions.sa.yes")  # back to your Yes: its grey box is 287
    ui.click("decisions.pic.open_tip", tip=True, within=ui.words("decisions.pic.grey"))
    ui.wait("edit.sel.their_face", timeout=60000)
    ui.shot("edit.selected.287", "The grey box, 287, in the editor", wait=1500)
    ui.press("r", on="edit.sel.their_face")
    ui.wait("decisions.piece.all")
    ui.shot("decisions.piece.287", "R on box 287: your No, and three rules' answers", wait=1000)
    ui.click("decisions.sa.no + decisions.rule", text=rule("decisions.sa.no"), nth=2)
    ui.shot("decisions.rule.weak", "A weak match, never asked: the rule's No")


PATHS = [first_open, same_no, same_yes, editor_rules, scan_replay]
