/* Run with node --test. Exercise rendered DOM and the actual editor callbacks. */
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
class Element {
  constructor(tag) { this.tagName=tag; this.children=[]; this.attrs={}; this.textContent=""; this.events={}; this.classes=new Set(); this.classList={toggle:(key,on)=>{if(on)this.classes.add(key);else this.classes.delete(key);},contains:key=>this.classes.has(key)}; }
  appendChild(child) { assert.notEqual(child,this); this.children.push(child); return child; }
  append(...children) { children.forEach(c=>this.appendChild(c)); }
  replaceChildren(...children) { this.children=[]; this.append(...children); }
  setAttribute(key,value) { this.attrs[key]=String(value); }
  addEventListener(key,fn) { this.events[key]=fn; }
  insertRow() { return this.appendChild(new Element("tr")); }
  insertCell() { return this.appendChild(new Element("td")); }
  createTHead() { return this.appendChild(new Element("thead")); }
  createTBody() { return this.appendChild(new Element("tbody")); }
  focus() {} select() {}
}
function runtime(name, extra={}) {
  const context={window:{},document:{createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)}, console};
  Object.assign(context,extra);
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../../esdc/portal/static",name),"utf8"),context);
  return context;
}
function nodes(root, tag) { return [root,...root.children.flatMap(c=>nodes(c,tag))].filter(x=>x.tagName===tag); }
function text(root) { return root.textContent+root.children.map(text).join(" "); }

test("chart breaks NULL and absent-year gaps independently and preserves exact table values",()=>{
  const {window}=runtime("chart.js"),root=new Element("div");
  window.renderComparisonSeries(root,[
    {year:2020,approved:"9007199254740993.001",outlook:null},
    {year:2021,approved:null,outlook:"0.000"},
    {year:2023,approved:"2.000",outlook:"3.000"},
    {year:2024,approved:"4.000",outlook:null}
  ],"MSTBY");
  const paths=nodes(root,"path").map(p=>p.attrs.d);
  assert.equal(paths.length,2);
  assert.equal((paths[0].match(/M/g)||[]).length,2);
  assert.equal((paths[0].match(/L/g)||[]).length,1);
  assert.equal((paths[1].match(/M/g)||[]).length,2);
  assert.equal((paths[1].match(/L/g)||[]).length,0);
  assert.ok(text(root).includes("9007199254740993.001"));
  assert.equal(nodes(root,"circle").length,5);
});

test("empty text is distinct from NULL and never becomes a plotted zero",()=>{
  const {window}=runtime("chart.js"),root=new Element("div");
  window.renderComparisonSeries(root,[{year:2020,approved:"",outlook:null}],"BSCFY");
  assert.equal(nodes(root,"circle").length,0);
  assert.ok(text(root).includes("(empty string)"));
  assert.ok(text(root).includes("—"));
});

test("chart retains full series and handles constant large magnitudes",()=>{
  const {window}=runtime("chart.js"),root=new Element("div");
  window.renderComparisonSeries(root,Array.from({length:151},(_,i)=>({year:2000+i,approved:"100000000000000000.001",outlook:null})),"MSTBY");
  assert.equal(nodes(root,"circle").length,151);
  assert.equal(nodes(root,"tr").length,152);
  assert.ok(!JSON.stringify(root).match(/NaN|Infinity/));
});

test("editors use Tabulator list definitions and custom input success/cancel lifecycle",()=>{
  const {window}=runtime("editors.js");
  const columns=window.PortalEditors.columns({readonly:false,pk:["id"],creation_inputs:[{field:"predecessors",title:"Prior",ref:"pods",refValue:"pod_id_seq",refLabel:"pod_name"}],columns:[
    {field:"amount",title:"Amount",logical_type:"decimal",editable:true},
    {field:"enabled",title:"Enabled",logical_type:"boolean",editable:true},
    {field:"pod_id_seq",title:"POD",ref:"pods",refValue:"pod_id_seq",refLabel:"pod_name",editableOnNew:true},
    {field:"project_name",title:"Project",logical_type:"text",entityList:true,editable:true}
  ]},{pods:[{pod_id_seq:"9007199254740993",pod_name:"POD",approval_date:"2024-01-01",pod_letter_num:"001"}]},()=>{},()=>{});
  const cell={getValue:()=>"12345678901234567.123",getRow:()=>({getData:()=>({_new:false})})};
  let committed, cancelled=false;
  const input=columns[0].editor(cell,fn=>fn(),v=>{committed=v;},()=>{cancelled=true;});
  assert.equal(input.value,"12345678901234567.123");
  input.value=""; input.events.blur(); assert.equal(committed,"");
  const second=columns[0].editor(cell,fn=>fn(),()=>assert.fail("Escape committed"),()=>{cancelled=true;});
  second.events.keydown({key:"Escape",preventDefault(){},stopPropagation(){}}); second.events.blur(); assert.equal(cancelled,true);
  assert.equal(columns[1].editor,"list");
  assert.equal(columns[2].editor,"list");
  assert.equal(columns[2].editable(cell),false);
  assert.equal(columns[2].editable({getRow:()=>({getData:()=>({_new:true})})}),true);
  const params=typeof columns[4].editorParams==="function"?columns[4].editorParams(cell):columns[4].editorParams;
  assert.equal(params.multiselect,true);
  assert.equal(params.values[0].value,"9007199254740993");
  const entity=columns[3].editor({getValue:()=>'["Rokan", "Kampar"]'},fn=>fn(),()=>{},()=>{});
  assert.equal(entity.value,"Rokan; Kampar");
  const rendered=columns[0].formatter(cell); assert.notEqual(rendered,cell);
  assert.equal(rendered.textContent,"12345678901234567.123");
});


test("diagnostic paging retains the last page on failure and clears stale errors after retry", async()=>{
  const ids=Object.fromEntries(["chart-config","result-pod","result-date","result-component","result-apply","result-status","result-chart","result-diagnostics"].map(key=>[key,new Element("div")]));
  ids["chart-config"].textContent=JSON.stringify({components:[{value:"oil",label:"Oil",unit:"MSTBY"}]});
  let pageTwoAttempts=0;
  const page=(number)=>({source_generation:1,rows:[{code:"D"+number,message:"Page "+number}],total:201,page:number,size:200,last_page:2,basis:null});
  const workspace={refs:{pods:[{pod_id_seq:"1",pod_name:"POD",approval_date:"2024-01-01",pod_letter_num:"001"}]},states:new Map([["pod_monitoring_api",{generation:1}],["pod_forecast_api",{generation:1}]]),setFilters:async()=>true,
    json:async(url)=>{
      if(url.startsWith("/api/chart"))return{source_generation:1,series:[],unit:"MSTBY"};
      if(url.includes("page=2")){if(++pageTwoAttempts===1)throw new Error("temporary failure");return page(2);}
      return page(1);
    }};
  const document={getElementById:id=>ids[id],createElement:tag=>new Element(tag)};
  const {window}=runtime("results.js",{document,PortalWorkspace:workspace,AbortController,renderComparisonSeries(){}});
  await window.initResults();
  ids["result-pod"].value="1";ids["result-date"].value="2024-12-31";ids["result-component"].value="oil";
  await ids["result-apply"].events.click();
  const next=nodes(ids["result-diagnostics"],"button").find(b=>b.textContent==="Next diagnostics");
  await next.events.click();
  assert.ok(text(ids["result-diagnostics"]).includes("Page 1 of 2"));
  assert.ok(ids["result-status"].textContent.includes("could not be loaded"));
  assert.equal(ids["result-status"].classList.contains("error"),true);
  await next.events.click();
  assert.ok(text(ids["result-diagnostics"]).includes("Page 2 of 2"));
  assert.equal(ids["result-status"].classList.contains("error"),false);
  assert.ok(!ids["result-status"].textContent.includes("could not be loaded"));
});
