var crsr = document.querySelector("#cursor")
var blur = document.querySelector("#cursor-blur")
document.addEventListener("mousemove",function(dets){
    crsr.style.left = dets.x+"px"
    crsr.style.top = dets.y+"px"
     blur.style.left = dets.x-150+"px"
    blur.style.top = dets.y-150+"px"
})

gsap.to("#nav",{
    backgroundColor:"#000",
    height:"110px",
    duration:0.5,
    scrollTrigger:{
        trigger: "#nav",
        scroller:"body",
         //markers:true,
        start:"top -10%",
        end:"top -10%",
        scrub:1
    }
})

gsap.to("#main",{
    backgroundColor:"#000",
    scrollTrigger:{
        trigger:"main",
        scroller:"body",
        markers:false,
        start:"top -25%",
        end:"top -70%",
        scrub:2

    }
})
gsap.from("#checkup", {  
    y: 50,
    opacity: 0,
    duration: 1,
    scrollTrigger: {
        trigger: "#page2",
        scroller: "body",
        start: "top 75%",
        end: "top 70%",
        
        scrub: 3
    }
});

gsap.from("#about-us img, #about-us-in",{
    y:50,
    opacity:0,
    duration:1,
    stagger:0.4,
    scrollTrigger:{
        trigger:"#about-us",
        scroller:"body",
        start:"top 70%",
        end:"top 65%",
        scrub:3
    }
})
document.querySelectorAll(".img-container").forEach(container => {
    let text = container.querySelector(".overlay-text");
    let image = container.querySelector("img");

    container.addEventListener("mouseenter", () => {
        gsap.to(text, {
            opacity: 1,
            y: -10,  // Moves text slightly upward
            duration: 0.5,
            ease: "power2.out"
        });

        gsap.to(image, {
            filter: "blur(10px)",  // Apply blur effect
            scale: 1.1,  // Slight zoom-in effect
            duration: 0.5,
            ease: "power2.out"
        });
    });

    container.addEventListener("mouseleave", () => {
        gsap.to(text, {
            opacity: 0,
            y: 0,
            duration: 0.5,
            ease: "power2.out"
        });

        gsap.to(image, {
            filter: "blur(0px)",  // Remove blur effect
            scale: 1,  // Reset zoom
            duration: 0.5,
            ease: "power2.out"
        });
    });
});
