// add optional named arguments

#import "@preview/physica:0.9.1": *
#import "@preview/tablex:0.0.7": tablex, hlinex, vlinex, colspanx, rowspanx
#import "@preview/ctheorems:1.1.0": *
#show: thmrules
#import "@preview/codly:0.1.0": codly-init, codly, disable-codly
#show: codly-init.with()
#set quote(block: true)
#import "@preview/wrap-it:0.1.0": wrap-content, wrap-top-bottom
#import "@preview/showybox:2.0.1": showybox


#let apply-template(body, title:[ *Neural Networks相关文献阅读报告 \
  _Error bounds for approximations with deep ReLU networks_* #footnote[https://arxiv.org/abs/1610.01145]]/*an example*/,
  right_header:"statistical learning homework"/*an example*/,
left_header:"statistical learning homework"/*an example*/,
author:"Maythics"/*an example*/,
ID:"3220104133",
link:"http..."
) = {
  set page(
  margin: (left:3.2em,right:3.2em),
  header: [
  #smallcaps(left_header)
  #h(1fr)  #text(right_header)
  #move(dx: 0pt, dy: -6pt,line(length: 100%))
  ],
  paper: "a4",
  numbering:"-1-",
)


  align(center, text(17pt,title))

  grid(
  columns: (1fr),
  align(center)[
    Author: #text(author) \
    ID: #text(ID) \
  ],
)

  show heading.where(
  level: 1
): it => block(width: 100%)[
  #set align(center)
  #set text(16pt, weight: "bold")
  #smallcaps(it.body)
]

  show heading.where(
  level: 2
): it => text(
  size: 14pt,
  weight: "regular",
  style: "italic",
  [#h(0.4em)]+it.body,
)

  show: rest => columns(1, rest)

  body
}



#let blue_theorem(title,footer,body)=showybox(
  title: text(title),
  frame: (
    border-color: blue,
    title-color: blue.lighten(50%),
    body-color: blue.lighten(97%),
    footer-color: blue.lighten(80%)
  ),
  footer: text(footer)
)[ #body ]

#let dark_theorem(title,footer,body)= showybox(
  frame:(
    title-color: black.lighten(30%),
  ),
  footer-style: (
    sep-thickness: 0pt,
    align: right,
    color: black
  ),

  title: text(title),
  footer: text(footer)
)[
 #body
]

#let red_theorem(title,body)= showybox(
  frame: (
    border-color: red.darken(30%),
    title-color: red.darken(30%),
    radius: 0pt,
    thickness: 2pt,
    body-inset: 2em,
    dash: "densely-dash-dotted"
  ),
  title: text(title),
)[
  #body
]

#let theorem = thmbox("theorem", "Theorem", fill: rgb("#eeeeff"))
#let corollary = thmplain("corollary", "Corollary", base: "theorem", titlefmt: strong)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Example").with(numbering: none)
#let proof = thmplain(
  "proof", "Proof", base: "theorem",
  bodyfmt: body => [#body #h(1fr) $square$]
).with(numbering: none)

/*my quote function*/

#let Quotation(name,body) = block(
  above: 2em, stroke: 0.1em + blue,
  width: 100%, inset: 14pt
)[
  #set quote(block: true, attribution: [#name])
  #set text(font: "Noto Sans", fill: black)
  #place(
    top + left,
    dy: -6pt - 14pt, // Account for inset of block
    dx: 6pt - 14pt,
    block(fill: white, inset: 2pt)[#text(blue,size: 12pt)[*Quote*]]
  )
  #align(left)[#quote[#body]]
]

#let t= h(2em) //blank space for writing


#let codex(code, lang: none, size: 1em, border: true) = {
  if code.len() > 0 {
    if code.ends-with("\n") {
      code = code.slice(0, code.len() - 1)
    }
  } else {
    code = "// no code"
  }
  set text(size: size)
  align(left)[
    #if border == true {
      block(
        width: 100%,
        stroke: 0.5pt + luma(150),
        radius: 4pt,
        inset: 8pt,
      )[
        #raw(lang: lang, block: true, code)
      ]
    } else {
      raw(lang: lang, block: true, code)
    }
  ]
}

#t Examples of how to use my DIY functions:

#blue_theorem("Proposition 01","footer content")[this is the body of proposition]

#dark_theorem("Gauss's Law","powered by Maythics")[#t the Flux equals to the total charges contained in an enclosed surface, i.e. $ integral.double_Sigma vectorarrow(E)dot dd(vectorarrow(S)) = integral.triple_Omega rho dd(V) $]

#red_theorem("miracle")[this is a red theorem]

#Quotation("luxin")[“聘为浙大教师的学历要求起步就有‘有海外学习、科研经历并取得一定的学术成果的博士’。一个典型的晋升路径，如浙大有‘百人计划’，在六年内带领学生满足一定的教学要求，开展自己的实验室建设，并在这个平台上发表高水平的研究成果，三四年左右成为教授，之后经过院系、学部、学校的层层考核，升任长聘教授。”
]



