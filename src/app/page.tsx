"use client";

import { LoadingSkeleton } from "@/components/loading/LoadingSkeleton";
import dynamic from "next/dynamic";

const ChatUI = dynamic(() => import("@/components/ChatUI"), {ssr: false, loading: () => <LoadingSkeleton />})

export default function Home() {
  return (
    <>
      <ChatUI />
    </>
  );
}
